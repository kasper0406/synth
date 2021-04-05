const ui = import("./pkg");

const Synth = require("../synth_wasm/src/synth_wasm.js");

export function setup_audio() {
    console.log("Setting up audio hook...");

    var audioCode = `
        class CaptureProcessor extends AudioWorkletProcessor {
            constructor(...args) {
                super(...args);

                this.port.onmessage = (e) => {
                    this.buffers = e.data.map((buf) => new Float32Array(buf));
                    this.indexes = Array(this.buffers.length).fill(0);
                };
            }

            process (inputs, outputs, parameters) {
                for (let i = 0; i < inputs[0].length; i++) {
                    outputs[0][i].set(inputs[0][i]);

                    for (let j = 0; j < inputs[0][i].length; j++) {
                        this.buffers[i][this.indexes[i]] = inputs[0][i][j];
                        this.indexes[i] = (this.indexes[i] + 1) % this.buffers[i].length;
                    }
                }

                return true;
            }
        }

        registerProcessor("capture-processor", CaptureProcessor);
    `;

    const audioContext = new AudioContext();
    Synth.createNode(audioContext).then(synthNode => {
        synthNode.connect(audioContext.destination);
    });

    /*
    var blob = new Blob([audioCode], { type: 'text/javascript' });
    var workerUrl = URL.createObjectURL(blob);

    console.log("Filename", __filename);

    const audioContext = new AudioContext();
    audioContext.audioWorklet.addModule(workerUrl).then(_ => {
        const synthNode = new AudioWorkletNode(audioContext, "synth-processor");
        const captureNode = new AudioWorkletNode(audioContext, "capture-processor");

        synthNode.connect(captureNode);
        captureNode.connect(audioContext.destination);

        const captureMemory = new SharedArrayBuffer(250000);
        captureNode.port.postMessage([ captureMemory ]);
        
        synthNode.port.postMessage([ synth_wasm, audioContext.sampleRate ]);
    }); */
};

ui.then(ui => {
    ui.run();
});
