This is optimized df-v5 remote explorer.

## HOWTO

build:
RUSTFLAGS="-C target-cpu=native" cargo build --release


run:
PORT=7890   ./target/release/mimc-miner

