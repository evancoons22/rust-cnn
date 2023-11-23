pub mod methods;
pub mod activation;

use crate::methods::nn::{Network, Layer};
use crate::methods::linalg::Matrix;  
use crate::activation::activations::Activation;

fn main() {

    let mut network = Network::new();

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
        activation: Activation::Tanh,
    };

    network.add_layer(layer1);
    network.add_layer(layer2);

    let inputs = vec![2.0, 1.0];
    let outputs = network.forward(&inputs);

    let inputs = vec![8.0, 1.0];
    let outputs2 = network.forward(&inputs);

    println!("{:?}", outputs);
    println!("{:?}", outputs2);
}