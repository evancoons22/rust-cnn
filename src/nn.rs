use crate::dataloader::DataLoader;
use crate::linalg::*;
use crate::activation::*;
use crate::loss::*;
use rand::Rng;

#[derive(Debug, PartialEq)]
pub struct Layer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Matrix,
    pub biases: Vec<f64>,
    pub activation: Activation,
    pub activationdata: Vec<f64>,
    pub activationgrad: Vec<f64>,
}

impl Clone for Layer {
    fn clone(&self) -> Self {
        Layer { 
            input_size: self.input_size,
            output_size: self.output_size,
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            activation: self.activation.clone(),
            activationdata: self.activationdata.clone(),
            activationgrad: self.activationgrad.clone(),
        }
    }
}

impl Layer { 
    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        // applies network weights AND activates
        self.activationdata = self.activation.forward(&add(&(&self.weights * &inputs), &self.biases));
        self.activationdata.clone()
    }
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        Layer {
            input_size,
            output_size,
            weights: Matrix::rand(output_size, input_size),
            // generate random biases in between -1 and 1
            biases: (0..output_size).map(|_| rand::thread_rng().gen_range(-1.0..1.0)).collect(),
            activation,
            activationdata: vec![0.0; output_size],
            activationgrad: vec![0.0; output_size],
        }
    }

    pub fn weight_grad_backwards(&mut self, inputs: &Vec<f64>, agradnext: &Vec<f64>, loss: &LossFunction, alpha: f64) {
        let mut result = Matrix::new(self.output_size, self.input_size);
        //let activation = self.activationdata.clone();
        let actgrad = self.activation.backward(&inputs, loss);
        //eprintln!("actgrad: {:?}", actgrad);
        //eprintln!("weight dimensions: {:?}x{:?}", result.data.len(), result.data[0].len());
        //eprintln!("result dimensions: {:?}x{:?}", result.data.len(), result.data[0].len());
        //eprintln!("output size: {:?}", self.output_size);
        //eprintln!("input size: {:?}", self.input_size);
        for j in 0..self.output_size {
            for k in 0..self.input_size {
                //result.data[j][k] = actgrad[k] * activation[j] * agradnext[j];
                // print all dimensions to make sure they match up
                result.data[j][k] = actgrad[j] * inputs[k] * agradnext[j];
            }
        }
        //result
        self.weights = self.weights.clone() - alpha * result;
    }

    pub fn bias_grad_backwards(&mut self, inputs: &Vec<f64>, agradnext: &Vec<f64>, loss: &LossFunction, alpha: f64) {
        let mut result = vec![0.0; self.output_size];
        let zgrad = self.activation.backward(&inputs, loss);
        for j in 0..self.output_size {
            result[j] = zgrad[j] * agradnext[j];
        }
        //result
        //self.biases = self.biases - scalar_mul_vec(alpha, vec)
        self.biases = weight_update(alpha, self.biases.clone(), result);
        
    }


    pub fn activation_grad(&mut self, weights: Matrix, agradnext: &Vec<f64>, anext: Vec<f64>, loss: &LossFunction) {
        use crate::linalg::*;
        let mut result = vec![0.0; self.input_size];
        let zgradnext = self.activation.backward(&anext, loss);
        let tweights = transpose(weights.clone());
        let first = dot_product(&agradnext, &zgradnext);

        for k in 0..self.input_size {
            result[k] = tweights.data[k].iter().map(|&x| x * first).sum();
        }
        self.activationdata = result;
        //result
    }
}

#[derive(Debug, PartialEq)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub loss: LossFunction,
}

impl Network { 
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            loss: LossFunction::MSE,
        }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn add_layers(&mut self, layers: Vec<Layer>) {
        for layer in layers {
            self.layers.push(layer);
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = inputs.clone();
        for layer in &mut self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }


    pub fn classify(&self, outputs: &Vec<f64>) -> Vec<f64> {
        outputs.iter().map(|x| if *x > 0.5 { 1.0 } else { 0.0 }).collect()
    }

    pub fn backward(&mut self, inputs: &Vec<f64>, y_true: &Vec<f64>, alpha: f64) {
        // the last layer is special... calculate the gradients for the last activations
        let mut activationgrad: Vec<f64> = vec![];
        let index: usize = self.layers.len() - 1;

        let outputs = self.layers[index].activationdata.clone();

        activationgrad.append(&mut self.loss.backward(&outputs, &y_true));

        // set the activation of the last layer equal to activationgrad
        self.layers[index].activationgrad = activationgrad.clone();

        // go through the layers backwards
        for i in (0..self.layers.len()).rev() {

            // if next gradient is the end, use the last activation data, otherwise, use the next layer's activationgrad
            let agradnext = if i == self.layers.len() - 1 {
                activationgrad.clone()
            } else {
                self.layers[i + 1].activationgrad.clone()
            };

            // if the input is the first layer, use the inputs, otherwise, use the previous layer's activation data
            let input = match i {
                0 => inputs.clone(),
                _ => self.layers[i - 1].activationdata.clone(),
            };

            
            if i == self.layers.len() - 1 { 
                // activation grad for the last layer
                self.layers[i].activationgrad = agradnext.clone();
                self.layers[i].weight_grad_backwards(&input, &agradnext, &self.loss, alpha);
            } else {
                // activation grad for the other layers
                let nextweights = self.layers[i + 1].weights.clone();
                let anext = self.layers[i + 1].activationdata.clone();
                let athis = self.layers[i].activationdata.clone();

                //update activations
                self.layers[i].activation_grad(nextweights, &agradnext, anext, &self.loss);

                // bias update is similar to weights
                self.layers[i].bias_grad_backwards(&input, &athis, &self.loss, alpha);
                self.layers[i].weight_grad_backwards(&input, &athis, &self.loss, alpha);
            }

        }

        }

    pub fn train(&mut self, dataloader: &mut DataLoader, alpha: f64, epochs: usize, verbose: bool) {
        for e in 0..epochs {
            let mut loss: f64 = 0.0;
            for _ in 0..dataloader.num_batches {
                for i in 0..dataloader.batch_size {
                    let (batch_data, batch_labels) = dataloader.next_batch();
                    self.forward(&batch_data[i]);
                    self.backward(&batch_data[i], &batch_labels[i], alpha);
                    loss += self.loss.getloss(&self.layers[self.layers.len() - 1].activationdata, &batch_labels[i]);
                }

            }

            if verbose {
                let mut correct = 0.0;
                for i in 0..dataloader.data.len() {
                    let outputs = self.forward(&dataloader.data[i]);
                    let outputs = self.classify(&outputs);
                    if outputs == dataloader.labels[i] {
                        correct += 1.0;
                    }
                }
                eprintln!("epoch {:?} loss: {:?}, accuracy: {:?}", e, loss, correct / dataloader.data.len() as f64);
            }
        }
    }

    pub fn save_weights(&self, filename: &str) {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(filename).expect("Unable to create file");
        for layer in &self.layers {
            for row in &layer.weights.data {
                for col in row {
                    file.write_all(format!("{},", col).as_bytes()).expect("Unable to write data");
                }
            }
            file.write_all(b"\n").expect("Unable to write data");
        }
    }

    pub fn load_weights(&mut self, filename: &str) {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        let file = File::open(filename).expect("Unable to open file");
        let reader = BufReader::new(file);
        let mut weights: Vec<Vec<f64>> = Vec::new();
        for line in reader.lines() {
            let line = line.expect("Unable to read line");
            let mut row: Vec<f64> = Vec::new();
            //remove the last comma from line
            let line = line[..line.len() - 1].to_string();
            for col in line.split(",") {
                row.push(col.parse::<f64>().expect("Unable to parse float"));
            }
            weights.push(row);
        }
        let mut index = 0;
        for layer in &mut self.layers {

            for i in 0..layer.weights.nrows {
                for j in 0..layer.weights.ncols {
                    layer.weights.data[i][j] = weights[index][i * layer.weights.ncols + j];
                }
            }
            index += 1;
        }

    }

}

// other useful functions,
pub fn to_onehot(labels: Vec<Vec<f64>>, size: usize) -> Vec<Vec<f64>> {
    if labels[0].len() == size {
        // return labels if already one hot encoded
        return labels;
    }
    let mut result = Vec::new();
    for label in labels {
        let mut onehot = vec![0.0; size];
        for i in 0..size {
            if label[0] == i as f64 {
                onehot[i] = 1.0;
            }
        }
        result.push(onehot);
    }
    result
}

pub fn classify(outputs: &Vec<f64>) -> Vec<f64> {
    outputs.iter().map(|x| if *x > 0.5 { 1.0 } else { 0.0 }).collect()
}

pub fn accuracy(y_true: &Vec<Vec<f64>>, y_pred: &Vec<Vec<f64>>) -> f64 {
    let mut correct = 0.0;
    for i in 0..y_true.len() {
        if y_true[i] == y_pred[i] {
            correct += 1.0;
        }
    }
    correct / y_true.len() as f64
}

pub fn weight_update(alpha: f64, previous: Vec<f64>, gradient: Vec<f64>) -> Vec<f64> {
    let mut result = vec![0.0; previous.len()];
    for i in 0..previous.len() {
        result[i] = previous[i] - alpha * gradient[i];
    }
    return result;
        
}

#[cfg(test)]
mod tests {
    use crate::linalg::*;
    use crate::nn::{Layer, Network};

    #[test]
    fn test_layer_forward() { 
        use crate::activation::Activation;
        let mut layer = Layer { 
            input_size: 2,
            output_size: 2,
            weights: Matrix { 
                nrows: 2,
                ncols: 2,
                data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            },
            biases: vec![0.0, 0.0],
            activation: Activation::None,
            activationdata: vec![0.0, 0.0],
            activationgrad: vec![0.0, 0.0],
        };

        let inputs = vec![1.0, 2.0];
        let outputs = vec![5.0, 11.0];
        let layer_outputs = layer.forward(&inputs);
        assert_eq!(layer_outputs.clone(), outputs.clone());
        let layer2 = layer.forward(&layer_outputs);
        assert_eq!(layer2, vec![27.0, 59.0]);
    }

    #[test]
    fn network_forward_twice() { 
        use crate::activation::Activation;
        use crate::loss::LossFunction;
        let layer = Layer { 
            input_size: 2,
            output_size: 2,
            weights: Matrix { 
                nrows: 2,
                ncols: 2,
                data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            },
            biases: vec![0.0, 0.0],
            activation: Activation::None,
            activationdata: vec![0.0, 0.0],
            activationgrad: vec![0.0, 0.0],
        };

        let inputs = vec![1.0, 2.0];
        let mut network = Network { layers: vec![layer.clone(), layer.clone()], loss: LossFunction::MSE};
        assert_eq!(network.forward(&inputs), vec![27.0, 59.0]);

        }

    #[test]
    fn layer_forward_new() { 
        use crate::activation::Activation;
        let mut layer = Layer::new(2, 2, Activation::None);
        layer.weights = Matrix { 
            nrows: 2,
            ncols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        // add 1 layer to the network
        let mut network = Network::new();
        network.add_layer(layer.clone());
        let inputs = vec![1.0, 2.0];
        let outputs = layer.forward(&inputs);
        assert_eq!(outputs.len(), 2);
    }


    #[test]
    fn class_network_test() {
        use crate::nn::Network;
        use crate::nn::Layer;
        use crate::activation::Activation;
        use crate::loss::*;
        use crate::dataloader::DataLoader;

        let mut network = Network::new();
        network.loss = LossFunction::CrossEntropy;

        network.add_layers(vec![
            Layer::new(2, 4, Activation::Relu),
            Layer::new(4, 2, Activation::Relu),
            Layer::new(2, 1, Activation::Sigmoid),
        ]);

        let mut dataloader = DataLoader::new(vec![
                                             vec![1.0, 1.0],
                                             vec![0.0, 0.0],
                                             vec![0.0, 1.0],
                                             vec![1.0, 0.0],],
                                        vec![
                                            vec![1.0],
                                            vec![1.0],
                                            vec![0.0],
                                            vec![0.0],], 1, false);



        //let initial_loss = network.forward(&dataloader.data[0]);
        let initial_loss = network.loss.getloss(&network.layers[network.layers.len() - 1].activationdata, &dataloader.labels[0]);

        network.train(&mut dataloader, 0.006, 100, false);

        let final_loss = network.loss.getloss(&network.layers[network.layers.len() - 1].activationdata, &dataloader.labels[0]);

        //eprintln!("initial network output: {:?}\n, final network output: {:?}", initial, last);
        eprintln!("initial loss: {:?}\n, final loss: {:?}", initial_loss, final_loss);
        //eprintln!("true output: {:?}\n", dataloader.labels[0]);

        //asser that initial loss is greater than final loss
        assert!(initial_loss > final_loss);

    }

}

