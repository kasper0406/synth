#![recursion_limit="256"]

extern crate yew;
extern crate wasm_bindgen;
extern crate console_error_panic_hook;
extern crate web_sys as ws;

use wasm_bindgen::prelude::*;
use console_error_panic_hook::set_once as set_panic_hook;

use yew::prelude::*;
use yew_router::prelude::*;

mod routes;
mod app;
mod welcome;

use routes::SynthRoutes;

pub struct SynthRouter {}

impl Component for SynthRouter {
    type Message = ();
    type Properties = ();

    fn create(_: Self::Properties, _: ComponentLink<Self>) -> Self {
        SynthRouter {}
    }

    fn change(&mut self, _: Self::Properties) -> bool {
        false
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        false
    }

    fn view(&self) -> Html {
        html! {
            <div style="-webkit-user-select: none; cursor: default;">
                <Router<SynthRoutes>
                    render = Router::render(|switch: SynthRoutes| {
                        match switch {
                            SynthRoutes::Welcome => html!{ <welcome::Welcome /> },
                            SynthRoutes::Synth => html!{ <app::App /> }
                        }
                    })
                    redirect = Router::redirect(|route: Route| {
                        SynthRoutes::Welcome
                    })
                />
            </div>
        }
    }
}

#[wasm_bindgen]
pub fn run() {
    set_panic_hook();

    yew::start_app::<SynthRouter>();
}
