
## A feed forward neural network in rust

An extremely lightweight, feed forward neural network. Define a network with ```network::new()```, specify layers and their weights, then run the network.  
**dependencies:**
- [rand](https://crates.io/crates/rand)

This network is compiled to wasm in the target/ directory, meant to be run in the browser. 

Created for Machine Learning (Math 156) at UCLA

### A simple example

```rust
pub mod methods;
pub mod activation;

use crate::methods::nn::{Network, Layer};
use crate::methods::linalg::Matrix;
use crate::activation::activations::Activation;
use crate::methods::loss::*;


fn main() {
    
    // define inputs as a vector. ex:
    let inputs: Vec<f64> = vec![1.0, 2.0];

    // initialize network
    let mut network = Network::new();

    // define layers -- input, output, weights, biases, activation
    let layer1 = Layer {
        input_size: 2,
        output_size: 3,
        weights: Matrix::rand(3, 2),
        biases: vec![0.0; 3],
        activation: Activation::None,
    };

    let layer2 = Layer {
        input_size: 3,
        output_size: 2,
        weights: Matrix::rand(2, 3),
        biases: vec![0.0; 2],
        activation: Activation::Sigmoid,
    };

    // add these layers to the network
    network.add_layer(layer1);
    network.add_layer(layer2);

    //define loss
    network.loss = LossFunction::CrossEntropy;

    let outputs = network.forward(&inputs);

    // can get loss 
    let y_true = vec![0.0, 1.0];
    let loss = LossFunction::CrossEntropy;
    println!("{:?}", loss.getloss(&outputs, &y_true));
    
    //print outputs
    outputs
}
```

