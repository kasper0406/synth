use yew::prelude::*;
use yew_router::prelude::*;

use crate::routes::SynthRoutes;

pub struct Welcome {

}

impl Component for Welcome {
    type Message = ();
    type Properties = ();

    fn create(_: Self::Properties, _: ComponentLink<Self>) -> Self {
        Welcome {}
    }

    fn change(&mut self, _: Self::Properties) -> bool {
        false
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        false
    }

    fn view(&self) -> Html {
        html! {
            <>
                <h1>{ "Welcome to TubeSynth! "}</h1>
                <p>{ "This is an online synthesizer made for lulz :-)" }</p>
                <RouterButton<SynthRoutes> route=SynthRoutes::Synth>{ "Play!" }</RouterButton<SynthRoutes>>
            </>
        }
    }
}
