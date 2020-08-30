build:
    cd src/synth/ && cargo build --release
    cd src/ui/ && yarn build
    cd src/app/ && cargo build

run:
    just build
    cd src/app/ && cargo run
