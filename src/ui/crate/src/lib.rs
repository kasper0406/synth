extern crate yew;
extern crate wasm_bindgen;
extern crate console_error_panic_hook;

use wasm_bindgen::prelude::*;
use console_error_panic_hook::set_once as set_panic_hook;

mod app;

use app::App;

#[wasm_bindgen]
pub fn run() {
    set_panic_hook();

    yew::start_app::<App>();
}
