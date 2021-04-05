build:
    cd src/synth/ && cargo build --release
    cd src/synth_wasm && wasm-pack --verbose build --target no-modules
    cd src/ui/ && yarn build
    cd src/app/ && cargo build

start:
    just build
    cd src/ui/ && yarn start
