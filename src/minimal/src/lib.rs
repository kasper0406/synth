extern crate wasm_bindgen;
extern crate console_error_panic_hook;
extern crate web_sys as ws;

use wasm_bindgen::prelude::*;
use console_error_panic_hook::set_once as set_panic_hook;

#[wasm_bindgen]
pub fn synth_callback() {
    ws::console::log_1(&"Inside synth callback in Rust!".into());
}
