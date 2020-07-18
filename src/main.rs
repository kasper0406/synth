extern crate portaudio;

mod fft;
use num::complex::Complex;
use num::Zero;

use portaudio as pa;
use std::thread;
use std::time;
use std::f64::consts::PI;

use gnuplot as plot;
use gnuplot::AxesCommon;

const SAMPLE_RATE: f64 = 48_000.0;
const FRAMES_PER_BUFFER: u32 = 256;
const CHANNELS: usize = 2;
const INTERLEAVED: bool = true;
const VOLUME: f32 = 0.2;
const LIMIT: f32 = 0.3;

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
    // waveform: Vec<f64>,
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

struct Synth {
    wave: Box<Wave>,
    phase: f64
}

impl Synth {
    fn new() -> Synth {
        // let wave = MemorizedWave::new(&SineWave {}, 512);

        // let wave = MemorizedWave::new(&SawtoothWave {}, 512);
        let wave = MemorizedWave::new(&AliasedWave::new(&SawtoothWave {}, 16), 512);

        Synth { wave: Box::new(wave), phase: 0f64 }
    }

    fn callback(&mut self, args: pa::stream::OutputCallbackArgs<f32>) -> pa::stream::CallbackResult {
        let frames = args.frames;
        let buffer = args.buffer;

        let target = 262f64;

        for i in 0..frames {
            buffer[i * 2] = self.wave.sample(self.phase) as f32;
            buffer[i * 2 + 1] = self.wave.sample(self.phase) as f32;

            self.phase += target / SAMPLE_RATE;
            if self.phase >= 1f64 {
                self.phase -= 1f64;
                if self.phase < 0f64 {
                    self.phase = 0f64;
                }
            }

            buffer[i * 2] *= VOLUME;
            buffer[i * 2 + 1] *= VOLUME;

            // Clipping
            if buffer[i * 2] > LIMIT {
                buffer[i * 2] = LIMIT;
            }
            if buffer[i * 2 + 1] > LIMIT {
                buffer[i * 2 + 1] = LIMIT;
            }
        }

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

fn main() {
    /*
    plot(&[
        &SineWave {},
        &MemorizedWave::new(&SineWave {}, 256),
        &AliasedWave::new(&SineWave {}, 8),
    ]); */

    plot(&[
        &SquareWave {},
        &MemorizedWave::new(&SquareWave {}, 256),
        &AliasedWave::new(&SquareWave {}, 32),
    ]);

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

    let mut synth = Synth::new();

    let mut stream = pa.open_non_blocking_stream(settings, move |args| synth.callback(args)).unwrap();
    stream.start().unwrap();

    thread::sleep(time::Duration::from_secs(2));

    stream.stop().unwrap();
}
