use yew_router::{prelude::*, Switch};

#[derive(Debug, Switch, Clone)]
pub enum SynthRoutes {
    #[to = "/"]
    Welcome,

    #[to = "/synth"]
    Synth,
}
