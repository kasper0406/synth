// import { synth_callback } from "../crate/Cargo.toml"

class SynthProcessor extends AudioWorkletProcessor {
    process (inputs, outputs, parameters) {
        const output = outputs[0];
        // synth_callback(output);
        console.log(output);
        debugger;
        return true;
    }
}

registerProcessor("synth-processor", SynthProcessor);
