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
    pub zdata: Vec<f64>,
    pub activationgrad: Vec<f64>,
    pub previous_layer: Option<Box<Layer>>,
    pub next_layer: Option<Box<Layer>>,
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
            zdata: self.zdata.clone(),
            activationgrad: self.activationgrad.clone(),
            previous_layer: self.previous_layer.clone(),
            next_layer: self.next_layer.clone(),
        }
    }
}

impl Layer { 
    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        // applies network weights AND activates
        self.zdata = add(&(&self.weights * &inputs), &self.biases);
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
            zdata: vec![0.0; output_size],
            activationgrad: vec![0.0; output_size],
            previous_layer: None,
            next_layer: None,
        }
    }

    pub fn weight_grad_backwards(&mut self, inputs: &Vec<f64>, loss: &LossFunction, alpha: f64) {
        let mut result = Matrix::new(self.output_size, self.input_size);
        let zgrad = self.activation.backward(&self.zdata, loss);
        for j in 0..self.output_size {
            for k in 0..self.input_size {
                result.data[j][k] = zgrad[j] * inputs[k] * self.activationgrad[j];
            }
        }

        self.weights = self.weights.clone() - alpha * result;
    }

    pub fn bias_grad_backwards(&mut self, loss: &LossFunction, alpha: f64) {
        // bias grad is similar to weight grad, but only 1d, and inputs are always 1
        let mut result = vec![0.0; self.output_size];
        let zgrad = self.activation.backward(&self.zdata, loss);
        for j in 0..self.output_size {
            result[j] = zgrad[j] * self.activationgrad[j];
        }

        self.biases = weight_update(alpha, self.biases.clone(), result);
    }


    pub fn activation_grad_backwards(&mut self, weights: Matrix, agradnext: &Vec<f64>, anext: Vec<f64>, loss: &LossFunction) {
        use crate::linalg::*;
        let mut result = vec![0.0; self.output_size];
        let zgradnext = self.activation.backward(&anext, loss);
        let tweights = transpose(weights.clone());
        let actgradnext = elementwise_mul(&agradnext, &zgradnext);

        for k in 0..self.output_size {
            result[k] = dot_product(&tweights.data[k], &actgradnext);
        }

        self.activationgrad = result;
    }
}

#[derive(Debug, PartialEq)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub loss: LossFunction,
    pub accuracy: f64,
}

impl Network { 
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            loss: LossFunction::MSE,
            accuracy: 0.0,
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

        // the last layer is special... calculate the gradients for the last activations with loss function
        let index: usize = self.layers.len() - 1; // last layer
        let outputs = self.layers[index].activationdata.clone();
        self.layers[index].activationgrad = self.loss.backward(&outputs, &y_true);


        // go through the layers backwards
        for i in (0..self.layers.len()).rev() {

            let input = match i {
                0 => inputs.clone(),
                _ => self.layers[i - 1].activationdata.clone(),
            };


            if i == self.layers.len() - 1 { 
                // last layer update (no activation grad)
                self.layers[i].weight_grad_backwards(&input, &self.loss, alpha);
                self.layers[i].bias_grad_backwards(&self.loss, alpha);
            } else {
                // activation grad for the other layers
                let nextweights = self.layers[i + 1].weights.clone();
                let anext = self.layers[i + 1].activationdata.clone();
                let agradnext = self.layers[i + 1].activationgrad.clone();

                //update
                self.layers[i].activation_grad_backwards(nextweights, &agradnext, anext, &self.loss);
                self.layers[i].bias_grad_backwards(&self.loss, alpha);
                self.layers[i].weight_grad_backwards(&input, &self.loss, alpha);
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

            let mut correct = 0.0;
            for i in 0..dataloader.data.len() {
                let outputs = self.forward(&dataloader.data[i]);
                let outputs = self.classify(&outputs);
                if outputs == dataloader.labels[i] {
                    correct += 1.0;
                }
            }
            if verbose {
                eprintln!("epoch {:?} loss: {:?}, accuracy: {:?}", e, loss, correct / dataloader.data.len() as f64);
            }
            self.accuracy = correct / dataloader.data.len() as f64;
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
            zdata: vec![0.0, 0.0],
            activationgrad: vec![0.0, 0.0],
            previous_layer: None,
            next_layer: None,
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
            zdata: vec![0.0, 0.0],
            activationgrad: vec![0.0, 0.0],
            previous_layer: None,
            next_layer: None,
        };

        let inputs = vec![1.0, 2.0];
        let mut network = Network { layers: vec![layer.clone(), layer.clone()], loss: LossFunction::MSE, accuracy: 0.0 };
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
    fn class_network_train() {
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
                                            vec![0.0],], 2, false);



        let initial_loss = network.loss.getloss(&network.layers[network.layers.len() - 1].activationdata, &dataloader.labels[0]);

        network.train(&mut dataloader, 0.006, 100, false);

        let final_loss = network.loss.getloss(&network.layers[network.layers.len() - 1].activationdata, &dataloader.labels[0]);

        eprintln!("initial loss: {:?}\n, final loss: {:?}", initial_loss, final_loss);

        //asser that initial loss is greater than final loss
        assert!(initial_loss >= final_loss);

    }

}

