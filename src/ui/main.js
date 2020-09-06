export function setup_audio() {
    console.log("Setting up audio hook...");

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
        const whiteNoiseNode = new AudioWorkletNode(audioContext, "synth-processor");
        whiteNoiseNode.connect(audioContext.destination);
    });
};

import("./pkg").then(rust => {
    console.log("Initializing rust code");
    rust.run();
});
