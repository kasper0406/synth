use yew::prelude::*;

use ws::AudioContext;
use ws::BlobPropertyBag;
use ws::Blob;
use ws::Url;
use ws::AudioWorkletNode;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;

pub struct App {

}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "synth"])]
    fn setup_audio();
}

impl Component for App {
    type Message = ();
    type Properties = ();

    fn create(_: Self::Properties, _: ComponentLink<Self>) -> Self {
        setup_audio();

        App {}
    }

    fn change(&mut self, _: Self::Properties) -> bool {
        false
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        false
    }

    fn view(&self) -> Html {
        html! {
            <h1>{ "Hello, World from Rust xD!" }</h1>
        }
    }
}
