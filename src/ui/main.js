// const rust = import("./pkg");
const minimalLoader = import("../minimal/pkg/minimal")

console.log("Defining functions...");

export function setup_audio() {
    console.log("Setting up audio hook...");

    var audioCode = `
        console.log('Inside audio setup code');

        class SynthProcessor extends AudioWorkletProcessor {
            constructor(...args) {
                super(...args);
                console.log("Inside SynthProcessor...");

                this.port.onmessage = (e) => {
                    let wasmBytes = e.data;
                    console.log("Received WASM bytes", wasmBytes);

                    var importObject = {
                        imports: {

                        }
                    }
                    WebAssembly.instantiate(wasmBytes, importObject).then(rust => {
                        console.log("Loaded rust wasm xD");
                        this.rust = rust.instance.exports;
                    });
                };
            }

            process (inputs, outputs, parameters) {
                if (this.rust !== undefined) {
                    this.rust.synth_callback(outputs);
                }

                return true;
            }
        }

        registerProcessor("synth-processor", SynthProcessor);
    `;

    // minimal.init();

    var blob = new Blob([audioCode], { type: 'text/javascript' });
    var workerUrl = URL.createObjectURL(blob);

    console.log("Filename", __filename);
    let wasmLoader = fetch("da8817554716fd3bb361.module.wasm");

    const audioContext = new AudioContext();
    audioContext.audioWorklet.addModule(workerUrl).then(_ => {
        const synthNode = new AudioWorkletNode(audioContext, "synth-processor");
        synthNode.connect(audioContext.destination);

        wasmLoader.then(response => response.arrayBuffer())
            .then(bytes => {
                synthNode.port.postMessage(bytes);
            })
    });
};
