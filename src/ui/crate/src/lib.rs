extern crate yew;
extern crate wasm_bindgen;
extern crate console_error_panic_hook;
extern crate web_sys as ws;

use wasm_bindgen::prelude::*;
use console_error_panic_hook::set_once as set_panic_hook;
use ws::AudioContext;

mod app;

use app::App;

#[wasm_bindgen]
pub fn run() {
    set_panic_hook();

    let sampleRate = 41800.0;
    const bufferSize: usize = 41800;

    let ctx = ws::AudioContext::new().unwrap();

    println!("Test here!");

    let buffer = ctx.create_buffer(2, bufferSize as u32, sampleRate).unwrap();
    let mut channel0 = buffer.get_channel_data(0).unwrap();
    let mut channel1 = buffer.get_channel_data(1).unwrap();

    for i in 0..bufferSize {
        let sample = (2.0 * std::f64::consts::PI * 440.0 * (i as f64) / 41800.0).sin() as f32;
        println!("Sample: {}", sample);
        channel0[i] = sample;
        channel1[i] = sample;
    }

    let buffer_src = ctx.create_buffer_source().unwrap();
    buffer_src.set_buffer(Some(&buffer));
    buffer_src.start().unwrap();

    yew::start_app::<App>();
}
