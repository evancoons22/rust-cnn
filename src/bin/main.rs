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
        Layer::new(2, 1, Activation::Softmax),
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

    //eprintln!("initial network output: {:?}\n, final network output: {:?}", initial, last);
    eprintln!("initial loss: {:?}\n, final loss: {:?}", initial_loss, final_loss);
    //eprintln!("true output: {:?}\n", dataloader.labels[0]);

}

