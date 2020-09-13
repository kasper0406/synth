const rust = import("./pkg");
const minimal = import("../minimal/pkg/minimal")

console.log("Defining functions...");
export function hi() {
    console.log("Hello!");
}

export function setup_audio() {
    console.log("Setting up audio hook...");

    var audioCode = `
        console.log('Inside audio setup code');

        class SynthProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                import("./pkg").then(rust => {
                    console.log("Loaded rust module in audio worklet");
                });
            }

            process (inputs, outputs, parameters) {
                // const output = outputs[0];
                window.m_rust.synth_callback();
                // console.log(output);

                // debugger;

                return true;
            }
        }

        registerProcessor("synth-processor", SynthProcessor);
    `;

    // minimal.init();

        /*
    var blob = new Blob([audioCode], { type: 'text/javascript' });
    var workerUrl = URL.createObjectURL(blob);

    console.log("Filename", __filename);

    const audioContext = new AudioContext();
    audioContext.audioWorklet.addModule(workerUrl).then(_ => {
        const whiteNoiseNode = new AudioWorkletNode(audioContext, "synth-processor");
        whiteNoiseNode.connect(audioContext.destination);
    });*/
};

rust.then(rust => {
    console.log("Initializing rust code");

    // TODO(knielsen): Consider passing this through the run function instead
    //                 or somehow set it with a setter function on the rust module
    // window.m_rust = {};
    // window.m_rust.synth_callback = rust.synth_callback;

    rust.run();
});
