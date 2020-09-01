use yew_router::{prelude::*, Switch};

#[derive(Debug, Switch, Clone)]
pub enum SynthRoutes {
    #[to = "/welcome"]
    Welcome,

    #[to = "/synth"]
    Synth,
}
