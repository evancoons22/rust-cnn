pub mod methods;
pub mod activation;

use crate::methods::nn::{Network, Layer};
use crate::methods::linalg::Matrix;
use crate::activation::activations::Activation;

#[no_mangle]
pub extern "C" fn run_network(inputs: Vec<f64>) -> Vec<f64> {
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

    // let inputs = vec![2.0, 1.0];
    let outputs = network.forward(&inputs);

    //println!("{:?}", outputs);

    //let y_pred = vec![0.001, 0.001, 0.998];
    //let y_true = vec![0.0, 0.0, 1.0];
    //let loss = LossFunction::CrossEntropy;


    //println!("{:?}", loss.getloss(&y_pred, &y_true));
    
    outputs
}
