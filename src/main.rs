#![feature(drain_filter,div_duration)]

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

use std::collections::VecDeque;

use gnuplot as plot;
use gnuplot::AxesCommon;

const SAMPLE_RATE: f64 = 48_000.0;
const FRAMES_PER_BUFFER: u32 = 16;
const CHANNELS: usize = 2;
const INTERLEAVED: bool = true;
const VOLUME: f32 = 0.3;
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

const OUTPUT_EPSILON: f64 = 0.05f64;
const NUM_PHASES: usize = 128;

struct PhaseManager {
    phases: [f64; NUM_PHASES],

    delta_indices: Vec<usize>,
    delta_values: [f64; NUM_PHASES],
}

impl PhaseManager {
    fn new() -> PhaseManager {
        PhaseManager { 
            phases: [0f64; NUM_PHASES],

            delta_indices: Vec::with_capacity(NUM_PHASES),
            delta_values: [0f64; NUM_PHASES],
        }
    }

    fn increment(&mut self, idx: usize, delta: f64) {
        if self.delta_values[idx] != 0f64 {
            assert!(self.delta_values[idx] == delta, "Expected delta between signals to be identical. This may lead to weird sounds!");
        } else {
            self.delta_values[idx] = delta;
            self.delta_indices.push(idx);
        }
    }

    fn apply(&mut self) {
        for idx_ref in &self.delta_indices {
            let idx = *idx_ref;
            self.phases[idx] = (self.phases[idx] + self.delta_values[idx]).fract();
            self.delta_values[idx] = 0f64;
        }
        self.delta_indices.clear();
    }

    fn get_phase(&self, idx: usize) -> f64 {
        self.phases[idx]
    }

    fn is_beginning(&self, idx: usize, epsilon: f64) -> bool {
        self.get_phase(idx) < epsilon || self.get_phase(idx) > 1f64 - epsilon
    }
}

struct MaterializedSignal {
    frequency: f64,
    wave: MemorizedWave,
    phase_index: usize,

    is_playing: bool,
    should_stop: bool,
    is_removable: bool,
}

impl MaterializedSignal {
    fn new(frequency: f64, wave: MemorizedWave, phase_index: usize) -> MaterializedSignal {
        MaterializedSignal {
            frequency,
            wave,
            phase_index,

            is_playing: false,
            should_stop: false,
            is_removable: true
        }
    }

    #[inline(always)]
    fn sample(&mut self, phases: &mut PhaseManager, sample_idx: u64, rate: f64) -> f64 {
        if self.is_removable && self.should_stop {
            return 0f64;
        }

        let phase_jump = self.frequency / rate;
        phases.increment(self.phase_index, phase_jump);

        let is_phase_beginning = phases.is_beginning(self.phase_index, phase_jump / 2f64);
        let next_sample = self.wave.sample(phases.get_phase(self.phase_index));

        if self.should_stop && is_phase_beginning {
            self.is_removable = true;
            self.is_playing = false;
            return 0f64;
        }

        if !self.is_playing && !is_phase_beginning {
            self.is_removable = true;
            return 0f64;
        }

        self.is_playing = true;
        self.is_removable = false;
        
        next_sample
    }
}

impl Drop for MaterializedSignal {
    fn drop(&mut self) {
        // println!("Dropping materialized signal at freq {}", self.frequency);
    }
}

struct MaterializedSignals {
    signals: Vec<MaterializedSignal>,
}

impl MaterializedSignals {
    fn signal_stop(&mut self) {
        for signal in &mut self.signals {
            signal.should_stop = true;
        }
    }

    fn is_playback_finished(&self) -> bool {
        self.signals.iter().all(|signal| signal.is_removable)
    }
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
    orphan_signals: VecDeque<MaterializedSignals>,
    phases: PhaseManager,

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
    fn new(is_running: Arc<AtomicBool>, signal_receiver: mpsc::Receiver<MaterializedSignals>) -> Synth {
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
            orphan_signals: VecDeque::with_capacity(50),
            phases: PhaseManager::new(),
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
            if self.orphan_signals.len() == self.orphan_signals.capacity() {
                // We need to accept some popping and get rid of the oldest orphaned signals
                // If this happen we should increase the orphaned_signals queue.
                self.gc_reclaim_send.send(self.orphan_signals.pop_back().unwrap()).unwrap();
                panic!("Increase orphan_signals queue size!");
            }
            self.current_signal.signal_stop();
            std::mem::swap(&mut self.current_signal, &mut new_signal);
            self.orphan_signals.push_front(new_signal);
        }

        while let Some(orphan) = self.orphan_signals.back() {
            if !orphan.is_playback_finished() {
                break;
            }
            self.gc_reclaim_send.send(self.orphan_signals.pop_back().unwrap()).unwrap();
        }

        for i in 0..frames {
            let mut output = 0f64;
            for signal in &mut self.current_signal.signals {
                output += signal.sample(&mut self.phases, self.sample_count, SAMPLE_RATE);
            }

            for orphan in &mut self.orphan_signals {
                for signal in &mut orphan.signals {
                    output += signal.sample(&mut self.phases, self.sample_count, SAMPLE_RATE);
                }
            }

            self.phases.apply();

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
    phase_index: usize,
}

impl Signal {
    fn materialize(&self) -> MaterializedSignal {
        MaterializedSignal::new(self.frequency, self.wave.materialize(), self.phase_index)
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
            phase_index: signal.phase_index,
        }
    }
}

struct AmplitudeTransformer {
    amplitude: f64
}

impl Filter for AmplitudeTransformer {
    fn apply(&self, frequency: f64) -> f64 {
        self.amplitude
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
            phase_index: signal.phase_index,
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
            phase_index: signal.phase_index,
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
            phase_index: signal.phase_index,
        }
    }
}

struct EnvelopeFunction {
    transform_supplier: Box<Fn(Duration) -> Arc<dyn SignalTransformer>>, // Function defined on [0;duration] interval
    duration: Duration, // Domain of the envelope function
}

struct Envelope {
    attack: Option<EnvelopeFunction>,
    decay: Option<EnvelopeFunction>,
    sustain: Option<EnvelopeFunction>,
    release: Option<EnvelopeFunction>
}

impl Envelope {

    fn constant(transformer: Arc<dyn SignalTransformer>) -> Envelope {
        let transformer_sustain = transformer.clone();
        let transformer_release = transformer.clone();

        Envelope {
            attack: None,
            decay: None,
            sustain: Some(EnvelopeFunction {
                duration: Duration::from_millis(1),
                transform_supplier: Box::new(move |_duration| transformer_sustain.clone()),
            }),
            release: Some(EnvelopeFunction {
                duration: Duration::from_secs(3600),
                transform_supplier: Box::new(move |_duration| transformer_release.clone()),
            }),
        }
    }

    fn linear_amplitude(attack_duration: Duration, attack_amp: f64,
                        decay_duration: Duration, decay_amp: f64,
                        release_duration: Duration) -> Envelope {
        let interpolate = |current_duration: Duration, phase_duration: Duration, from, to| {
            from + (to - from) * current_duration.div_duration_f64(phase_duration)
        };

        Envelope {
            attack: Some(EnvelopeFunction {
                duration: attack_duration,
                transform_supplier: Box::new(move |duration| Arc::new(AmplitudeTransformer {
                    amplitude: interpolate(duration, attack_duration, 0f64, attack_amp)
                }))
            }),
            decay: Some(EnvelopeFunction {
                duration: decay_duration,
                transform_supplier: Box::new(move |duration| Arc::new(AmplitudeTransformer {
                    amplitude: interpolate(duration, decay_duration, attack_amp, decay_amp)
                }))
            }),
            sustain: Some(EnvelopeFunction {
                duration: Duration::from_millis(1),
                transform_supplier: Box::new(move |_duration| Arc::new(AmplitudeTransformer {
                    amplitude: decay_amp
                }))
            }),
            release: Some(EnvelopeFunction {
                duration: release_duration,
                transform_supplier: Box::new(move |duration| Arc::new(AmplitudeTransformer {
                    amplitude: interpolate(duration, release_duration, decay_amp, 0f64)
                }))
            }),
        }
    }

    /**
     * Current semantics:
     *  - Jump straight to release phase when the key is released
     *  - Repeat sustain as long as key is pressed
     *  - If no sustain is specified, release will be played immediately after decay
     */
    fn evaluate_at_time(&self, press_timestamp: Instant, possible_release_timestamp: Option<Instant>) -> Option<Arc<dyn SignalTransformer>> {
        if let Some(release_timestamp) = possible_release_timestamp {
            if let Some(release_func) = &self.release {
                let time_since_release = release_timestamp.elapsed();
                // println!("Inside release function: {:?} < {:?} = {:?}", time_since_release, release_func.duration, time_since_release < release_func.duration);
                if time_since_release < release_func.duration {
                    return Some((release_func.transform_supplier)(time_since_release));
                }
            }
            return None;
        }

        let mut elapsed_function_time = Duration::new(0, 0);
        let time_since_press = press_timestamp.elapsed();

        if let Some(attack_func) = &self.attack {
            if attack_func.duration > time_since_press {
                return Some((attack_func.transform_supplier)(time_since_press));
            }
            elapsed_function_time += attack_func.duration;
        }

        if let Some(decay_func) = &self.decay {
            if decay_func.duration + elapsed_function_time > time_since_press {
                return Some((decay_func.transform_supplier)(time_since_press - elapsed_function_time));
            }
            elapsed_function_time += decay_func.duration;
        }

        if let Some(sustain_func) = &self.sustain {
            let cycle_time = ((time_since_press - elapsed_function_time).as_nanos() as u64) % (sustain_func.duration.as_nanos() as u64);
            return Some((sustain_func.transform_supplier)(Duration::from_nanos(cycle_time)));
        }
        if let Some(release_func) = &self.release {
            if release_func.duration + elapsed_function_time > time_since_press {
                return Some((release_func.transform_supplier)(time_since_press - elapsed_function_time));
            }
        }

        None
    }
}

struct SignalPipeline {
    envelopes: Vec<Envelope>,
}

struct SignalIdentifier {
    signal: Arc<Signal>,
    press_timestamp: Instant,
    possible_release_timestamp: Option<Instant>,
    key_number: u16,
    velocity: u16,
}

impl SignalPipeline {
    fn new(envelopes: Vec<Envelope>) -> SignalPipeline {
        SignalPipeline { envelopes }
    }

    fn process_one(&self, sid: &SignalIdentifier) -> Option<MaterializedSignal> {
        let mut transformed_signal: &Signal = &sid.signal;

        let mut owner;
        for envelope in self.envelopes.iter() {
            if let Some(transformer) = envelope.evaluate_at_time(sid.press_timestamp, sid.possible_release_timestamp) {
                owner = transformer.transform(&transformed_signal);
                transformed_signal = &owner;
            } else {
                return None
            }
        }

        Some(MaterializedSignal::new(
            transformed_signal.frequency,
            transformed_signal.wave.materialize(),
            transformed_signal.phase_index
        ))
    }
}

enum MidiEvent {
    NoteOff { key_number: u16, timestamp: Instant },
    NoteOn { key_number: u16, velocity: u16, timestamp: Instant  },
}

fn build_waves() -> Vec<Arc<Signal>> {
    let mut result = Vec::with_capacity(88);
    for i in 0..89 {
        let frequency = 440.0f64 * ((((i as f64) - 69f64 + 20f64) as f64) / 12f64).exp2();
        result.push(Arc::new(Signal {
            frequency: frequency,
            wave: AliasedWave::new(&SineWave {}, 1024),
            phase_index: i,
        }))
    }
    result
}

fn synth_event_loop(is_running: Arc<AtomicBool>,
                    key_receiver: mpsc::Receiver<MidiEvent>,
                    sender: mpsc::Sender<MaterializedSignals>) {
    // Setup pipeline
    let pipeline = SignalPipeline::new(vec![
        Envelope::linear_amplitude(
            Duration::from_millis(300), 1.0,
            Duration::from_millis(200), 0.8,
            Duration::from_millis(100)
        ),
        // Envelope::constant(Arc::new(OvertoneGenerator { overtone_levels: vec![0.0, 0.57, 0.69, 0.34, 0.34, 0.28, 0.34, 0.46, 0.24, 0.34, 0.24 ] })),
        // Envelope::constant(Arc::new(BandwidthExpand { expand_exponent: 5 })),
        // Envelope::constant(Arc::new(DiffuseTransform { iterations: 2, leak_amount: 0.05 })),
        Envelope::constant(Arc::new(LowpassFilter { cutoff: SAMPLE_RATE / 2.0 })),
    ]);

    // Pre-compute possible waves and signals
    let waves = build_waves();

    // Sounds processing loop
    // TODO(knielsen): Add some metrics of how much is being processed
    let mut signal_ids: Vec<SignalIdentifier> = vec![];
    while (*is_running).load(Ordering::SeqCst) {
        while let Ok(event) = key_receiver.try_recv() {
            match event {
                MidiEvent::NoteOn { key_number, velocity, .. } => {
                    if signal_ids.iter().any(|sid| sid.key_number == key_number) {
                        // TODO(knielsen): Probably a bad idea having io here, but probably fine for now as this is an edge case
                        println!("Warning: Key that is already on is pressed again. Ignoring...");
                    } else {
                        assert!(key_number >= 21 && key_number <= 108, "Expected piano MIDI key range");
                        let wave_number = (key_number - 21) as usize;
                        signal_ids.push(SignalIdentifier {
                            key_number,
                            signal: waves[wave_number].clone(),
                            velocity,
                            press_timestamp: Instant::now(),
                            possible_release_timestamp: None,
                        });
                    }
                },
                MidiEvent::NoteOff { key_number, .. } => {
                    if let Some(sid) = signal_ids.iter_mut().find(|sid| sid.key_number == key_number) {
                        sid.possible_release_timestamp = Some(Instant::now());
                    }
                }
            }
        }

        let mut materialized_signals = vec![];
        signal_ids.drain_filter(|sid| {
            let processed = pipeline.process_one(&sid);
            if let Some(materialized) = processed {
                materialized_signals.push(materialized);
                return false;
            }
            true
        });

        sender.send(MaterializedSignals { signals: materialized_signals }).unwrap();

        thread::sleep(Duration::from_millis(10));
    }
}

fn play_something(midi_sender: mpsc::Sender<MidiEvent>) {
    for i in 0..5 {
        midi_sender.send(MidiEvent::NoteOn {
            key_number: 50 + i,
            velocity: 80,
            timestamp: Instant::now(),
        }).unwrap();

        thread::sleep(Duration::from_millis(2000));

        midi_sender.send(MidiEvent::NoteOff {
            key_number: 50 + i,
            timestamp: Instant::now(),
        }).unwrap();

        thread::sleep(Duration::from_millis(1000));
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

    let is_running = Arc::new(AtomicBool::new(true));

    // TODO(knielsen): Consider taking sample rate from output_info?
    let params = pa::stream::Parameters::new(output_device, CHANNELS as i32, INTERLEAVED, output_info.default_low_output_latency);
    let settings = pa::stream::OutputSettings::new(params, SAMPLE_RATE, FRAMES_PER_BUFFER);

    let (signal_sender, signal_receiver) = mpsc::channel();
    let mut synth = Synth::new(is_running.clone(), signal_receiver);

    let mut stream = pa.open_non_blocking_stream(settings, move |args| synth.callback(args)).unwrap();
    stream.start().unwrap();

    let (midi_event_sender, midi_event_receiver) = mpsc::channel();

    let event_loop_running = is_running.clone();
    let synth_loop = thread::spawn(move || {
        synth_event_loop(event_loop_running, midi_event_receiver, signal_sender)
    });

    play_something(midi_event_sender);

    is_running.store(false, Ordering::SeqCst);
    synth_loop.join().unwrap();

    stream.stop().unwrap();
}
