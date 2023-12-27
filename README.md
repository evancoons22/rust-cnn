
## A feed forward neural network in rust

An extremely lightweight, feed forward neural network. Define a network with ```network::new()```, specify layers and their weights, then run the network.  
**dependencies:**
- [rand](https://crates.io/crates/rand)

This network is compiled to wasm in the target/ directory, meant to be run in the browser. 

Created for Machine Learning (Math 156) at UCLA

### A simple example

```rust
fn main() { 
    //import crates   
    use crate::nn::Network;
    use crate::nn::Layer;
    use crate::activation::Activation;
    use crate::loss::*;

    //define inputs
    let inputs = vec![1.0, 1.0];
    let target = vec![2.0, 2.0];

    //define layers
    let mut layer1 = Layer::new(2, 4, Activation::Relu);
    layer1.biases = vec![0.0, 0.0, 0.0, 0.0];
    let mut layer2 = Layer::new(4, 2, Activation::Relu);
    layer2.biases = vec![0.0, 0.0];

    //define network ... add layers
    let mut network = Network::new();
    network.add_layer(layer1.clone());
    network.add_layer(layer2.clone());

    //train
    for i in 0..100 {
        network.forward(&inputs);
        network.backward(&inputs, &target, 0.004);
        println!("loss on round {}: {:?}", i, network.loss.getloss(&network.layers[1].activationdata, &target));
    }

    println!("network output: {:?}", network.forward(&inputs);

}
```

