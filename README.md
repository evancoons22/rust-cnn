
## A feed forward neural network in rust

An extremely lightweight, feed forward neural network. Define a network with ```network::new()```, specify layers and their weights, then run the network.  
**dependencies:**
- [rand](https://crates.io/crates/rand)

This network is compiled to wasm in the target/ directory, meant to be run in the browser. 

Created for Machine Learning (Math 156) at UCLA

### A simple example

```rust

// import crates
use rustnn::nn::Network;
use rustnn::nn::Layer;
use rustnn::activation::Activation;
use rustnn::loss::*;
use rustnn::dataloader::DataLoader;

fn main() {

    // initialize a new network and define loss
    let mut network = Network::new();
    network.loss = LossFunction::CrossEntropy;

    // add layers to network
    network.add_layers(vec![
        Layer::new(2, 4, Activation::Relu),
        Layer::new(4, 2, Activation::Relu),
        Layer::new(2, 1, Activation::Sigmoid),
    ]);

    // load data
    let dataloader = DataLoader::new(vec![
                                         vec![1.0, 1.0],
                                         vec![0.0, 0.0],
                                         vec![0.0, 1.0],
                                         vec![1.0, 0.0],], // data inputs
                                     vec![
                                         vec![1.0],
                                         vec![1.0],
                                         vec![0.0],
                                         vec![0.0]], // data labels
                                     1,  // batch size
                                     false); // shuffle



    network.train(&dataloader, 0.006, 100, false); // 0.006 learning rate, 100 epochs, verbose = false
    network.save_weights("weights.txt");
}
```

