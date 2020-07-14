extern crate portaudio;

use portaudio as pa;
use std::thread;
use std::time;
use std::f64::consts::PI;

const SAMPLE_RATE: f64 = 48_000.0;
const FRAMES_PER_BUFFER: u32 = 256;
const CHANNELS: usize = 2;
const INTERLEAVED: bool = true;
const VOLUME: f32 = 0.2;

struct Synth {
    waveform: Vec<f64>,
    phase: f64
}

impl Synth {
    fn new() -> Synth {
        const FIDELITY: usize = 500;
        let mut waveform = vec![0f64; FIDELITY];
        for i in 0..FIDELITY {
            waveform[i] = (((i as f64) / (FIDELITY as f64)) * 2f64 * PI).sin();
        }

        Synth { waveform, phase: 0f64 }
    }

    fn callback(&mut self, args: pa::stream::OutputCallbackArgs<f32>) -> pa::stream::CallbackResult {
        let frames = args.frames;
        let buffer = args.buffer;

        let target = 262f64;

        for i in 0..frames {
            buffer[i * 2] = self.waveform[(self.phase * self.waveform.len() as f64) as usize] as f32;
            buffer[i * 2 + 1] = self.waveform[(self.phase * self.waveform.len() as f64) as usize] as f32;

            self.phase += target / SAMPLE_RATE;
            if self.phase >= 1f64 {
                self.phase -= 1f64;
                if self.phase < 0f64 {
                    self.phase = 0f64;
                }
            }

            buffer[i * 2] *= VOLUME;
            buffer[i * 2 + 1] *= VOLUME;

            if buffer[i * 2] > VOLUME {
                buffer[i * 2] = 0f32;
            }
            if buffer[i * 2 + 1] > VOLUME {
                buffer[i * 2 + 1] = 0f32;
            }
        }

        pa::stream::CallbackResult::Continue
    }
}

fn main() {
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
