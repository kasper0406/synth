// const wasm = import("./pkg");
// const minimalLoader = import("../minimal/pkg/minimal")

console.log("Defining functions...");

export function setup_audio() {
    console.log("Setting up audio hook...");

    var audioCode = `
        console.log('Inside audio setup code');

        class SynthProcessor extends AudioWorkletProcessor {
            #wasmImportObj = (() => {
                let wasm;

                function setWasm(new_wasm) {
                    wasm = new_wasm;
                }

                class Synth {
                    static __wrap(ptr) {
                        const obj = Object.create(Synth.prototype);
                        obj.ptr = ptr;
                        return obj;
                    }

                    free() {
                        wasm.__wbindgen_free(this.buffer_ptr, this.buffer_len * 4);

                        const ptr = this.ptr;
                        this.ptr = 0;
                        wasm.__wbg_synth_free(ptr);
                    }

                    constructor(sample_rate) {
                        var ret = wasm.synth_new(sample_rate);
                        const instance = Synth.__wrap(ret);

                        instance.buffer_len = 128;
                        instance.buffer_ptr = wasm.__wbindgen_malloc(instance.buffer_len * 4);
                        instance.buffer = new Float32Array(wasm.memory.buffer)
                                            .subarray(instance.buffer_ptr / 4, instance.buffer_ptr / 4 + instance.buffer_len);

                        return instance;
                    }

                    generate(output) {
                        wasm.synth_generate(this.ptr, this.buffer_ptr, this.buffer_len);
                        output.set(this.buffer);
                    }
                }

                let cachedMemory = null;
                function getMemory() {
                    if (cachedMemory == null || cachedMemory.buffer !== wasm.memory.buffer) {
                        cachedMemory = new Uint8Array(wasm.memory.buffer);
                    }
                    return cachedMemory;
                }

                function decodeUtf8(bytes) {
                    return decodeURIComponent(bytes.reduce((p, c) => {
                        if (typeof p === "string") {
                            return p + "%" + c.toString(16)
                        } else {
                            return "%" + p.toString(16) + "%" + c.toString(16);
                        }
                    }));
                }

                function getStringFromWasm(ptr, len) {
                    const bytes = getMemory().subarray(ptr, ptr + len);
                    const utf8decoded = decodeUtf8(bytes);
                    return utf8decoded;
                }

                const imports = {};
                imports.wbg = {};
                imports.wbg.__wbg_log_9686041a8580e419 = function(arg0, arg1) {
                    console.log(getStringFromWasm(arg0, arg1));
                };
                imports.wbg.__wbindgen_throw = function(arg0, arg1) {
                    throw new Error(getStringFromWasm(arg0, arg1));
                };

                imports.control = {};
                imports.control.setWasm = setWasm;

                imports.Synth = Synth;

                return imports;
            })();

            constructor(...args) {
                super(...args);
                console.log("Inside SynthProcessor...");

                this.port.onmessage = (e) => {
                    const wasmBytes = e.data[0];
                    const sampleRate = e.data[1];

                    WebAssembly.instantiate(wasmBytes, this.#wasmImportObj).then(wasm => {
                        console.log("Loaded wasm wasm xD");
                        this.wasm = wasm.instance.exports;
                        this.#wasmImportObj.control.setWasm(this.wasm);

                        this.wasm.test_rust();
                        this.synth = new this.#wasmImportObj.Synth(sampleRate);
                    });
                };
            }

            process (inputs, outputs, parameters) {
                if (this.synth !== undefined) {
                    this.synth.generate(outputs[0][0]);
                }

                return true;
            }
        }

        registerProcessor("synth-processor", SynthProcessor);
    `;

    var blob = new Blob([audioCode], { type: 'text/javascript' });
    var workerUrl = URL.createObjectURL(blob);

    console.log("Filename", __filename);
    let wasmLoader = fetch("minimal.wasm");
    let wasmImportLoader = fetch("minimal.js");

    const audioContext = new AudioContext();
    audioContext.audioWorklet.addModule(workerUrl).then(_ => {
        const synthNode = new AudioWorkletNode(audioContext, "synth-processor");
        synthNode.connect(audioContext.destination);

        Promise.all([ wasmLoader, wasmImportLoader ])
            .then((responses) => Promise.all([ responses[0].arrayBuffer(), responses[1].text() ]))
            .then((wasm) => {
                synthNode.port.postMessage([ wasm[0], audioContext.sampleRate ]);
            });
    });
};
