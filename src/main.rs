extern crate portaudio;

mod fft;
use num::complex::Complex;
use num::Zero;

use portaudio as pa;
use std::thread;
use std::time;
use std::f64::consts::PI;

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
        let mut waveform = vec![0f64; num_samples];
        for i in 0..num_samples {
            waveform[i] = wave.sample((i as f64) / (num_samples as f64));
        }
        let coefficients = fft::naive_dft(&waveform);

        AliasedWave { coefficients }
    }
}

impl Wave for AliasedWave {
    fn sample(&self, x: f64) -> f64 {
        let mut result = Complex::<f64>::zero();
        let n = self.coefficients.len();
        for k in 0..(n/2) {
            let exp = Complex::<f64>::new(0.0, 2.0 * PI * (k as f64) * x);
            result += self.coefficients[k] * exp.exp();
        }
        2f64 * result.re
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

type FrequencyResponse = Fn(f64) -> f64;

struct Signal {
    wave: AliasedWave,
    frequency: f64,
}

impl Signal {
    fn materialize(&self) -> MaterializedSignal {
        MaterializedSignal {
            frequency: self.frequency,
            wave: MemorizedWave::new(&self.wave, 1024)
        }
    }
}

/*
struct Filter {
    signals: Vec<Signals>,
    func: FrequencyResponse, // Frequency response map
}

impl Filter {
    fn apply(&self, signal: SignalDescription) -> SignalDescription {
       
    }
} */

fn test_synth(sender: mpsc::Sender<MaterializedSignals>) {
    let mut signals = vec![];
    for i in 0..100 {
        let wave = MemorizedWave::new(&SineWave {}, 512);
        signals.push(MaterializedSignal {
            frequency: 7902.13 / (i + 1) as f64,
            wave: wave
        });
    }

    let signal1 = MaterializedSignals {
        signals
    };
    sender.send(signal1);

    thread::sleep(time::Duration::from_secs(3));

    let wave2 = MemorizedWave::new(&SineWave {}, 512);
    let signal2 = MaterializedSignals {
        signals: vec![
            MaterializedSignal {
                frequency: 392.00,
                wave: wave2
            }
        ],
    };
    sender.send(signal2);

    thread::sleep(time::Duration::from_secs(3));
}

fn main() {
    /*
    plot(&[
        &SineWave {},
        &MemorizedWave::new(&SineWave {}, 256),
        &AliasedWave::new(&SineWave {}, 8),
    ]); */

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
        &AliasedWave::new(&SawtoothWave {}, 64),
    ]); */

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
