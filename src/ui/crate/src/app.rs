use yew::prelude::*;

pub struct App {

}

impl Component for App {
    type Message = ();
    type Properties = ();

    fn create(_: Self::Properties, _: ComponentLink<Self>) -> Self {
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
            <div style="-webkit-user-select: none; cursor: default;">
                <h1>{ "Hello, World from Rust xD!" }</h1>
            </div>
        }
    }
}
