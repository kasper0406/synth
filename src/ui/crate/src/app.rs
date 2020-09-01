use yew::prelude::*;

use ws::AudioContext;

pub struct App {

}

impl Component for App {
    type Message = ();
    type Properties = ();

    fn create(_: Self::Properties, _: ComponentLink<Self>) -> Self {
        let sampleRate = 41800.0;
        const bufferSize: usize = 41800;
    
        let ctx = ws::AudioContext::new().unwrap();
    
        ws::console::log_1(&"Running audio code!".into());
    
        let buffer = ctx.create_buffer(2, bufferSize as u32, sampleRate).unwrap();
        let mut channel0 = buffer.get_channel_data(0).unwrap();
        let mut channel1 = buffer.get_channel_data(1).unwrap();
    
        for i in 0..bufferSize {
            let sample = (2.0 * std::f64::consts::PI * 440.0 * (i as f64) / 41800.0).sin() as f32;
            channel0[i] = sample;
            channel1[i] = sample;
        }
    
        let buffer_src = ctx.create_buffer_source().unwrap();
        buffer_src.set_buffer(Some(&buffer));
        buffer_src.connect_with_audio_node(&ctx.destination());

        ws::console::log_1(&"Starting play now!".into());
        buffer_src.start().unwrap();

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
