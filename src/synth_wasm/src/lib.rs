extern crate wasm_bindgen;
// extern crate console_error_panic_hook;
// extern crate web_sys as ws;
// extern crate js_sys;

use wasm_bindgen::prelude::*;
// use console_error_panic_hook::set_once as set_panic_hook;
// use js_sys::SharedArrayBuffer;
// use js_sys::Float32Array;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn test_rust() -> f64 {
    log("Inside rust 😁");
    123.8
}

#[wasm_bindgen]
pub struct Synth {
    sample_rate: f64,
    phase: f64,
    samples: Vec<f32>,
}

#[wasm_bindgen]
impl Synth {
    pub fn new(sample_rate: f64) -> Synth {
        log("Initializing 🎹");

        let sample_size = 1024;
        let mut samples = Vec::with_capacity(sample_size);
        for i in 0 .. sample_size {
            let input = ((i as f64) / (sample_size as f64)) * 2f64 * std::f64::consts::PI;
            let value = input.sin() as f32;
            samples.push(value);
        }

        Synth {
            sample_rate,
            phase: 0.0,
            samples
        }
    }

    pub fn generate(&mut self, output: &mut [f32]) {
        let frequency = 261.62;

        for i in 0 .. output.len() {
            let idx = (self.phase * output.len() as f64) as usize;
            output[i] = self.samples[idx] * 0.2;
            self.phase = (self.phase + frequency / self.sample_rate).fract();
        }
    }
}
