import { run } from "../crate/Cargo.toml"
import fs from 'fs'

run();

// const audioWorkerBlob = fs.readFileSync(__dirname + "/synth.js", "utf-8");

var audioCode = `
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
`;

var blob = new Blob([audioCode], { type: 'text/javascript' });
var workerUrl = URL.createObjectURL(blob);

const audioContext = new AudioContext()
audioContext.audioWorklet.addModule(workerUrl).then(_ => {
    debugger;
    const whiteNoiseNode = new AudioWorkletNode(audioContext, "synth-processor");
    whiteNoiseNode.connect(audioContext.destination);
});
