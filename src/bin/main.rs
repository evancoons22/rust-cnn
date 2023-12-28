use rustnn::nn::Network;
use rustnn::nn::Layer;
use rustnn::activation::Activation;
use rustnn::loss::*;
use rustnn::dataloader::DataLoader;

fn main() {

    let mut network = Network::new();
    network.loss = LossFunction::CrossEntropy;

    network.add_layers(vec![
        Layer::new(2, 4, Activation::Relu),
        Layer::new(4, 2, Activation::Relu),
        Layer::new(2, 1, Activation::Relu),
    ]);

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



    network.train(&dataloader, 0.006, 100, false);
    network.save_weights("weights.txt");


    println!("network forward: {:?} ", network.forward(&dataloader.data[0]));

    let mut network2 = Network::new();
    network2.loss = LossFunction::CrossEntropy;
    network2.add_layers(vec![
        Layer::new(2, 4, Activation::Relu),
        Layer::new(4, 2, Activation::Relu),
        Layer::new(2, 1, Activation::Relu),
    ]);

    network2.load_weights("weights.txt");
    println!("network2 forward: {:?} ", network2.forward(&dataloader.data[0]));

}

