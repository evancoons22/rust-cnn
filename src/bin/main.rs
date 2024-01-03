use rustnn::nn::*;
use rustnn::activation::Activation;
use rustnn::loss::*;
use rustnn::dataloader::DataLoader;

fn main() {

    let mut network = Network::new();
    network.loss = LossFunction::MSE;

    network.add_layers(vec![
        Layer::new(2, 2, Activation::Relu),
        Layer::new(2, 1, Activation::Sigmoid),
    ]);

    let data = vec![ vec![1.0, 1.0], vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0],];
    let labels = vec![ vec![1.0], vec![1.0], vec![0.0], vec![0.0],];
    //let labels = to_onehot(labels, 0);

    let mut dataloader = DataLoader::new(data, labels, 1, false); // 1 = batch size and false = shuffle

    network.train(&mut dataloader, 0.01, 2, true);
    //network.save_weights("weights.txt");

    println!("network forward: {:?} ", network.forward(&dataloader.data[0]));

    //let mut network2 = Network::new();
    //network2.loss = LossFunction::MSE;
    //network2.add_layers(vec![
        //Layer::new(2, 4, Activation::Relu),
        //Layer::new(4, 2, Activation::Relu),
        //Layer::new(2, 1, Activation::Relu),
    //]);

    //network2.load_weights("weights.txt");
    //println!("network2 forward: {:?} ", network2.forward(&dataloader.data[0]));

}

