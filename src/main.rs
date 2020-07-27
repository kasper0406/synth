extern crate portaudio;

mod fft;
use num::complex::Complex;
use num::Zero;

use portaudio as pa;
use std::thread;
use std::time;
use std::f64::consts::PI;
use std::time::Instant;
use std::time::Duration;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::sync::mpsc;

use gnuplot as plot;
use gnuplot::AxesCommon;

const SAMPLE_RATE: f64 = 48_000.0;
const FRAMES_PER_BUFFER: u32 = 64;
const CHANNELS: usize = 2;
const INTERLEAVED: bool = true;
const VOLUME: f32 = 0.2;
const LIMIT: f32 = 0.5;

trait Wave {
    // x should be in range [0, 1)
    fn sample(&self, x: f64) -> f64;
}

struct SineWave {}

impl Wave for SineWave {
    fn sample(&self, x: f64) -> f64 {
        (x * 2f64 * PI).sin()
    }
}

struct SawtoothWave {}

impl Wave for SawtoothWave {
    fn sample(&self, x: f64) -> f64 {
        (x - 0.5) * 2f64
    }
}

struct SquareWave {}

impl Wave for SquareWave {
    fn sample(&self, x: f64) -> f64 {
        if x < 0.5 {
            1f64
        } else {
            -1f64
        }
    }
}

struct PianoWave {}

impl Wave for PianoWave {
    fn sample(&self, a: f64) -> f64 {
        let x = (a - 0.5) * 2.0;
        -((3.0 * PI * x).sin() / 4.0) + ((PI * x).sin() / 4.0) + ((3.0 as f64).sqrt() * (PI * x).cos()) / 2.0
    }
}

struct MemorizedWave {
    waveform: Vec<f64>,
}

impl MemorizedWave {
    fn new(wave: &Wave, num_samples: usize) -> MemorizedWave {
        let mut waveform = vec![0f64; num_samples];
        for i in 0..num_samples {
            waveform[i] = wave.sample((i as f64) / (num_samples as f64));
        }
        MemorizedWave { waveform }
    }
}

impl Wave for MemorizedWave {
    fn sample(&self, x: f64) -> f64 {
        self.waveform[(x * self.waveform.len() as f64) as usize]
    }
}

struct AliasedWave {
    coefficients: Vec<Complex<f64>>,
}

impl AliasedWave {
    fn new(wave: &Wave, num_samples: usize) -> AliasedWave {
        let mut waveform = vec![Complex::<f64>::zero(); num_samples];
        for i in 0..num_samples {
            waveform[i] = Complex::<f64>::new(wave.sample((i as f64) / (num_samples as f64)), 0.0);
        }
        let coefficients = fft::recursive_fft(&waveform);

        AliasedWave::from_coefficients(coefficients)
    }

    fn from_coefficients(coefficients: Vec<Complex<f64>>) -> AliasedWave {
        AliasedWave { coefficients }
    }

    fn materialize(&self) -> MemorizedWave {
        let samples = fft::inverse_dft(&self.coefficients, fft::recursive_fft);
        let waveform = samples.iter().map(|c| c.re).collect();

        MemorizedWave { waveform }
    }
}

struct MaterializedSignal {
    frequency: f64,
    wave: MemorizedWave,
}

impl MaterializedSignal {
    #[inline(always)]
    fn sample(&self, sample_idx: u64, rate: f64) -> f64 {
        let phase = ((sample_idx as f64) * self.frequency / rate).fract();
        self.wave.sample(phase)
    }
}

impl Drop for MaterializedSignal {
    fn drop(&mut self) {
        println!("Dropping materialized signal at freq {}", self.frequency);
    }
}

struct MaterializedSignals {
    signals: Vec<MaterializedSignal>,
}

macro_rules! join_thread {
    ($x:expr) => {
        if let Some(thread) = $x.take() {
            thread.join().unwrap();
        }
    };
}

struct CallbackStats {
    num_completed: Arc<AtomicUsize>,

    // All times in nanoseconds
    max_time: Arc<AtomicUsize>,
    combined_time: Arc<AtomicUsize>,
}

impl CallbackStats {
    fn new() -> CallbackStats {
        CallbackStats {
            num_completed: Arc::new(AtomicUsize::new(0)),
            max_time: Arc::new(AtomicUsize::new(0)),
            combined_time: Arc::new(AtomicUsize::new(0)),
        }
    }
}

struct Synth {
    sample_count: u64, // More than enough to never overflow

    // Signal handling
    signal_receiver: mpsc::Receiver<MaterializedSignals>,
    current_signal: MaterializedSignals,

    // Signal cleanup handling
    gc_thread: Option<JoinHandle<()>>,
    gc_reclaim_send: mpsc::Sender<MaterializedSignals>,

    // Callback monitoring handling
    callback_stats: CallbackStats,
    monitoring_thread: Option<JoinHandle<()>>,

    // Generic
    is_running: Arc<AtomicBool>,
}

impl Drop for Synth {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        join_thread!(self.gc_thread);
        join_thread!(self.monitoring_thread);
    }
}

impl Synth {
    fn new(signal_receiver: mpsc::Receiver<MaterializedSignals>) -> Synth {
        let is_running = Arc::new(AtomicBool::new(true));

        let gc_running = is_running.clone();
        let (gc_reclaim_send, gc_reclaim_recv) = mpsc::channel();
        let gc_thread = Some(thread::spawn(move || {
            while (*gc_running).load(Ordering::SeqCst) {
                if let Ok(garbage) = gc_reclaim_recv.recv_timeout(time::Duration::from_millis(250)) {
                    std::mem::drop(garbage);
                }
            }
        }));

        let callback_stats = CallbackStats::new();
        let monitoring_running = is_running.clone();
        let monitoring_cb_completed = callback_stats.num_completed.clone();
        let monitoring_combined_time = callback_stats.combined_time.clone();
        let monitoring_max_time = callback_stats.max_time.clone();
        let monitoring_thread = Some(thread::spawn(move || {
            let mut prev_time = time::Instant::now();
            while (*monitoring_running).load(Ordering::SeqCst) {
                thread::sleep(time::Duration::from_secs(1));
                // Theres a bit of data race in these numbers, but it will give a rough idea of performance
                let num_completed = monitoring_cb_completed.fetch_and(0, Ordering::Relaxed);
                let combined_time = monitoring_combined_time.fetch_and(0, Ordering::Relaxed);
                let max_time = monitoring_max_time.fetch_and(0, Ordering::Relaxed);

                let after_time = time::Instant::now();
                let duration = after_time.duration_since(prev_time);
                prev_time = after_time;

                println!("{} callbacks completed in {:?}, combined cb time {} ms, max cb execution {} us",
                    num_completed, duration, combined_time / 1000000, max_time / 1000);
            }
        }));

        let zero_signal = MaterializedSignals {
            signals: vec![]
        };

        Synth {
            sample_count: 0,
            signal_receiver,
            current_signal: zero_signal,
            is_running,
            gc_thread,
            gc_reclaim_send,
            callback_stats,
            monitoring_thread,
        }
    }

    fn callback(&mut self, args: pa::stream::OutputCallbackArgs<f32>) -> pa::stream::CallbackResult {
        let start_time = time::Instant::now();

        let frames = args.frames;
        let buffer = args.buffer;

        if let Ok(mut new_signal) = self.signal_receiver.try_recv() {
            std::mem::swap(&mut self.current_signal, &mut new_signal);
            self.gc_reclaim_send.send(new_signal).unwrap();
        }

        for i in 0..frames {
            let mut output = 0f64;
            for signal in &self.current_signal.signals {
                output += signal.sample(self.sample_count, SAMPLE_RATE);
            }

            buffer[i * 2] = output as f32;
            buffer[i * 2 + 1] = output as f32;

            buffer[i * 2] *= VOLUME;
            buffer[i * 2 + 1] *= VOLUME;

            /*
            // Clipping
            if buffer[i * 2] > LIMIT {
                buffer[i * 2] = LIMIT;
            }
            if buffer[i * 2 + 1] > LIMIT {
                buffer[i * 2 + 1] = LIMIT;
            } */

            self.sample_count += 1;
        }

        // Update monitoring
        let elapsed = start_time.elapsed().as_nanos() as usize;
        self.callback_stats.num_completed.fetch_add(1, Ordering::Relaxed);
        self.callback_stats.max_time.fetch_max(elapsed, Ordering::Relaxed);
        self.callback_stats.combined_time.fetch_add(elapsed, Ordering::Relaxed);

        pa::stream::CallbackResult::Continue
    }
}

struct Signal {
    wave: AliasedWave,
    frequency: f64,
}

impl Signal {
    fn materialize(&self) -> MaterializedSignal {
        MaterializedSignal {
            frequency: self.frequency,
            wave: self.wave.materialize()
        }
    }
}

trait SignalTransformer {
    fn transform(&self, signal: &Signal) -> Signal;
}

trait Filter: SignalTransformer {
    fn apply(&self, frequency: f64) -> f64;    
}

impl<T: Filter> SignalTransformer for T {
    fn transform(&self, signal: &Signal) -> Signal {
        let mut coefficients = signal.wave.coefficients.clone();

        let n = coefficients.len();
        for i in 0..(n / 2) {
            let scalar = self.apply(signal.frequency * (i as f64));
            coefficients[i] = scalar * coefficients[i];
            coefficients[n - i - 1] = scalar * coefficients[n - i - 1];
        }

        Signal {
            wave: AliasedWave::from_coefficients(coefficients),
            frequency: signal.frequency,
        }
    }
}

struct LowpassFilter {
    cutoff: f64,
}
impl Filter for LowpassFilter {
    fn apply(&self, frequency: f64) -> f64 {
        if frequency < self.cutoff {
            1f64
        } else {
            0f64
        }
    }
}

struct HighpassFilter {
    cutoff: f64,
}
impl Filter for HighpassFilter {
    fn apply(&self, frequency: f64) -> f64 {
        if frequency > self.cutoff {
            1f64
        } else {
            0f64
        }
    }
}

struct OvertoneGenerator {
    overtone_levels: Vec<f64>,
}
impl SignalTransformer for OvertoneGenerator {
    fn transform(&self, signal: &Signal) -> Signal {
        let n = signal.wave.coefficients.len();
        let mut coefficients = vec![Complex::<f64>::zero(); 2 * n];

        /*
        for i in 1..coefficients.len() {
            let tone_value = coefficients[i];
            for j in 0..self.overtone_levels.len() {
                let index = i * (j + 2);
                if index >= coefficients.len() {
                    break;
                }
                coefficients[index] += signal.wave.coefficients[i] * self.overtone_levels[j];
            }
        } */

        for i in 0..n {
            coefficients[2 * i] = signal.wave.coefficients[i];
        }

        let tone_value_1 = coefficients[2];
        let tone_value_2 = coefficients[2 * n - 3];

        coefficients[1] += tone_value_1 * self.overtone_levels[0];
        coefficients[2 * n - 2] += tone_value_2 * self.overtone_levels[0];

        for i in 2..n.min(self.overtone_levels.len()) {
            coefficients[2 * i] += tone_value_1 * self.overtone_levels[i - 1];
            coefficients[2 * n - (2 * i) - 1] += tone_value_2 * self.overtone_levels[i - 1];
        }

        Signal {
            wave: AliasedWave::from_coefficients(coefficients),
            frequency: signal.frequency / 2.0,
        }
    }
}

struct BandwidthExpand {
    expand_exponent: usize
}

impl SignalTransformer for BandwidthExpand {
    fn transform(&self, signal: &Signal) -> Signal {
        let expand_factor = 1 << self.expand_exponent;
        let mut coefficients = vec![Complex::<f64>::zero(); expand_factor * signal.wave.coefficients.len()];

        for i in 0..signal.wave.coefficients.len() {
            coefficients[expand_factor * i] = signal.wave.coefficients[i];
        }

        Signal {
            wave: AliasedWave::from_coefficients(coefficients),
            frequency: signal.frequency / (expand_factor as f64),
        }
    }
}

struct DiffuseTransform {
    iterations: usize,
    leak_amount: f64,
}

impl SignalTransformer for DiffuseTransform {
    fn transform(&self, signal: &Signal) -> Signal {
        let mut prev = signal.wave.coefficients.clone();
        let mut next = signal.wave.coefficients.clone();

        for i in 0..self.iterations {
            for j in 0..next.len() {
                let left_leak = if j > 0 { prev[j - 1] } else { Complex::<f64>::zero() };
                let right_leak = if j < prev.len() - 1 { prev[j + 1] } else { Complex::<f64>::zero() };
                next[j] = prev[j] + self.leak_amount * (left_leak + right_leak);
            }
            std::mem::swap(&mut prev, &mut next);
        }

        Signal {
            wave: AliasedWave::from_coefficients(prev),
            frequency: signal.frequency,
        }
    }
}

struct EnvelopeFunction {
    transform_supplier: Box<Fn(Duration) -> Box<dyn SignalTransformer>>, // Function defined on [0;duration] interval
    duration: Duration, // Domain of the envelope function
}

struct Envelope {
    attack: Option<EnvelopeFunction>,
    decay: Option<EnvelopeFunction>,
    sustain: Option<EnvelopeFunction>,
    release: Option<EnvelopeFunction>
}

impl Envelope {
    /**
     * Current semantics:
     *  - Jump straight to release phase when the key is released
     *  - Repeat sustain as long as key is pressed
     *  - If no sustain is specified, release will be played immediately after decay
     */
    fn evaluate_at_time(&self, press_timestamp: Instant, possible_release_timestamp: Option<Instant>) -> Option<Box<dyn SignalTransformer>> {
        if let Some(release_timestamp) = possible_release_timestamp {
            if let Some(release_func) = self.release {
                let time_since_release = release_timestamp.elapsed();
                if release_func.duration < time_since_release {
                    return Some((release_func.transform_supplier)(time_since_release));
                }
            }
            return None;
        }

        let mut elapsed_function_time = Duration::new(0, 0);
        let time_since_press = press_timestamp.elapsed();

        if let Some(attack_func) = self.attack {
            if attack_func.duration > time_since_press {
                return Some((attack_func.transform_supplier)(time_since_press));
            }
            elapsed_function_time += attack_func.duration;
        }

        if let Some(decay_func) = self.decay {
            if decay_func.duration + elapsed_function_time > time_since_press {
                return Some((decay_func.transform_supplier)(time_since_press - elapsed_function_time));
            }
            elapsed_function_time += decay_func.duration;
        }

        if let Some(sustain_func) = self.sustain {
            let cycle_time = ((time_since_press - elapsed_function_time).as_nanos() as u64) % (sustain_func.duration.as_nanos() as u64);
            return Some((sustain_func.transform_supplier)(Duration::from_nanos(cycle_time)));
        }
        if let Some(release_func) = self.release {
            if release_func.duration + elapsed_function_time > time_since_press {
                return Some((release_func.transform_supplier)(time_since_press - elapsed_function_time));
            }
        }

        None
    }
}

struct SignalPipeline {
    transformers: Vec<Box<Envelope>>,
}

impl SignalPipeline {
    fn new(transformers: Vec<Box<Envelope>>) -> SignalPipeline {
        SignalPipeline { transformers }
    }

    fn process_one(&self, signal: &Signal, time_since_press: f64, time_since_release: Option<f64>) -> Option<MaterializedSignal> {
        let mut transformed_signal = signal;

        let mut owner;
        for transformer in self.transformers.iter() {
            owner = transformer.transform(transformed_signal);
            transformed_signal = &owner;
        }

        Some(MaterializedSignal {
            frequency: transformed_signal.frequency,
            wave: transformed_signal.wave.materialize(),
        })
    }
}

fn test_synth(sender: mpsc::Sender<MaterializedSignals>) {
    let inputs = vec![
        Signal {
            frequency: 197.0,
            wave: AliasedWave::new(&SineWave {}, 512)
        },
    ];

    let mut signals = vec![];

    for input in inputs {
        let pipeline = SignalPipeline::new(vec![
            Box::new(OvertoneGenerator { overtone_levels: vec![0.34, 0.57, 0.69, 0.34, 0.34, 0.28, 0.34, 0.46, 0.24, 0.34, 0.24 ] }),
            Box::new(BandwidthExpand { expand_exponent: 5 }),
            Box::new(DiffuseTransform { iterations: 2, leak_amount: 0.05 }),
            Box::new(LowpassFilter { cutoff: SAMPLE_RATE / 2.0 }),
        ]);
        signals.push(pipeline.process_one(&input));
    }

    sender.send(MaterializedSignals { signals }).unwrap();

    thread::sleep(time::Duration::from_millis(2000));
}

fn plot(waves: &[&Wave]) {
    const SAMPLES: usize = 512;
    let mut sampled_waves = Vec::with_capacity(waves.len());

    for wave in waves {
        let mut x: Vec<f64> = Vec::with_capacity(SAMPLES);
        let mut y: Vec<f64> = Vec::with_capacity(SAMPLES);

        for i in 0..SAMPLES {
            let a = (i as f64) / (SAMPLES as f64);
            x.push(a);
            y.push(wave.sample(a));
        }

        sampled_waves.push((x, y));
    }

    let mut figure = plot::Figure::new();
    let axes = figure.axes2d()
        .set_title("Signal plot", &[])
        .set_x_label("Time", &[])
        .set_y_label("Amplitude", &[]);
    
    let colors = ["red", "web-green", "web-blue", "dark-orange", "orange", "dark-yellow"]
        .iter()
        .map(|color| plot::Color(*color))
        .cycle();

    for (i, ((x, y), color)) in sampled_waves.iter().zip(colors).enumerate() {
        axes.points(x, y, &[plot::Caption(&format!("Wave {}", i)), color]);
    }

    figure.show().unwrap();
}

fn main() {
    /*
    plot(&[
        &PianoWave {},
        &MemorizedWave::new(&PianoWave {}, 256),
        &AliasedWave::new(&PianoWave {}, 8),
    ]);
    */

    /*
    plot(&[
        &SquareWave {},
        &MemorizedWave::new(&SquareWave {}, 256),
        &AliasedWave::new(&SquareWave {}, 32),
    ]);*/

    /*
    plot(&[
        &SawtoothWave {},
        &MemorizedWave::new(&SawtoothWave {}, 256),
        &AliasedWave::new(&SawtoothWave {}, 256).materialize(),
    ]);*/

    /*
    let aliased_wave = AliasedWave::new(&SawtoothWave {}, 1024);
    let signal = Signal {
        frequency: 1.0,
        wave: aliased_wave,
    };
    let filtered = LowpassFilter { cutoff: 8.0 }.transform(&signal);

    plot(&[
        &signal.wave.materialize(),
        &filtered.wave.materialize(),
    ]);*/

    let pa = pa::PortAudio::new().unwrap();
    println!("PortAudio version: {}", pa.version());

    let output_device = pa.default_output_device().unwrap();
    let output_info = pa.device_info(output_device).unwrap();
    println!("Default output device info: {:#?}", &output_info);

    // TODO(knielsen): Consider taking sample rate from output_info?
    let params = pa::stream::Parameters::new(output_device, CHANNELS as i32, INTERLEAVED, output_info.default_low_output_latency);
    let settings = pa::stream::OutputSettings::new(params, SAMPLE_RATE, FRAMES_PER_BUFFER);

    let (signal_sender, signal_receiver) = mpsc::channel();
    let mut synth = Synth::new(signal_receiver);

    let mut stream = pa.open_non_blocking_stream(settings, move |args| synth.callback(args)).unwrap();
    stream.start().unwrap();

    test_synth(signal_sender);

    stream.stop().unwrap();
}
