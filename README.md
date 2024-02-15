# Candle BHOP

Basin hopping optimisation for candle achine learning models

## Installation

To use this crate a rust toolchain must be installed. This can be done by following the instructions at [rustup.rs](https://rustup.rs/).

## Running examples

There are two examples included with the crate, a multiplayer perceptron  and a convolutional autoencoder, both for MNIST.
This can be run by

```bash
cargo r -r --example mnist-mlp
```

or

```bash
cargo r -r --example autoencoder
```

To accelerate learning, cuda support can be enabled:

```bash
cargo r -r --example mnist-mlp --features cuda
```
