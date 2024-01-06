use rustnn::nn::*;
use rustnn::activation::Activation;
use rustnn::loss::*;
use rustnn::dataloader::DataLoader;

fn main() {

    // create network
    let mut network = Network::new();
    network.loss = LossFunction::CrossEntropy;
    network.add_layers(vec![
        Layer::new(2, 4, Activation::Relu),
        Layer::new(4, 1, Activation::Sigmoid),
    ]);

    // add data
    let data = vec![ vec![1.0, 1.0], vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![4.0, 4.0], vec![5.0, 5.0], vec![5.0, 4.0], vec![4.0, 5.0]];
    let labels = vec![ vec![1.0], vec![1.0], vec![1.0], vec![1.0], vec![0.0], vec![0.0], vec![0.0], vec![0.0]];


    let mut dataloader = DataLoader::new(data, labels, 1, false); // 1 = batch size and false = shuffle

    network.train(&mut dataloader, 0.1, 30, true); // learning rate, epochs, verbose
    //network.save_weights("weights.txt"); // can save weights to file

}

// mnist example

//let mut dataloader = DataLoader::new_csv("../data/mnist_test.csv", 0, 1, false); // 0 = label index, 1 = batch size and true = shuffle
//dataloader.normalize_data();                                                                                  
//dataloader.num_batches = 1;
//dataloader.labels_to_categorical(10); // 10 = number of classes                                                                                 
//network.add_layers(vec![
//Layer::new(784, 256, Activation::Relu),
//Layer::new(256, 128, Activation::Relu),
//Layer::new(128, 10, Activation::Softmax),
//]);
