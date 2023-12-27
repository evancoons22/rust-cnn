use crate::linalg::*;
use crate::activation::*;
use crate::loss::*;

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
            biases: vec![0.0; output_size],
            activation,
            activationdata: vec![0.0; output_size],
            activationgrad: vec![0.0; output_size],
        }
    }

    pub fn weight_grad_backwards(&self, inputs: &Vec<f64>, agradnext: &Vec<f64>) -> Matrix {
        let mut result = Matrix::new(self.output_size, self.input_size);
        let activation = self.activationdata.clone();
        let actgrad = self.activation.backward(&inputs, &LossFunction::MSE);
        //eprintln!("input size: {:?}", &self.input_size);
        //eprintln!("output size: {:?}", &self.output_size);
        //eprintln!("inputs len: {:?}", &inputs.len()); // 2
        //eprintln!("actgrad len: {:?}", &actgrad.len()); // 4
        //eprintln!("activation len: {:?}", &activation.len()); // 2
        //eprintln!("agradnext len: {:?}", &agradnext.len()); // 2
        for j in 0..self.output_size {
            for k in 0..self.input_size {
                result.data[j][k] = actgrad[k] * activation[j] * agradnext[j];
            }
        }
        result
    }

    pub fn bias_grad_backwards(&self, inputs: &Vec<f64>, agradnext: &Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; self.output_size];
        let actgrad = self.activation.backward(&inputs, &LossFunction::MSE);
        for j in 0..self.output_size {
            result[j] = actgrad[j] * agradnext[j];
        }
        result
    }


    pub fn activation_grad(&self, inputs: &Vec<f64>, weights: Matrix, agradnext: &Vec<f64>) -> Vec<f64> {
        use crate::linalg::*;
        let mut result = vec![0.0; self.input_size];
        //let actgrad = self.activation.backward(&inputs, &LossFunction::MSE);
        let actgrad = self.activation.backward(&self.activationdata, &LossFunction::MSE);
        let tweights = transpose(weights.clone());
        let first = dot_product(&agradnext, &actgrad);
        //eprintln!("actgrad len: {:?}", &actgrad.len());
        //eprintln!("actgradnext len: {:?}", &agradnext.len());
        //eprintln!("tweights ncols: {:?}", &tweights.ncols);
        //eprintln!("length of result: {:?}", &result.len());

        for k in 0..self.input_size {
            //result[k] += dot_product(&[dot_product(&tweights.data[k], &actgrad)], &agradnext);
            //result[k] += dot_product(&tweights.data[k], &add(&actgrad, &agradnext));
            //result[k] = dot_product(&tweights.data[k], &first);
            
            //result[k] = tweights[k] * first;
            //iterate over tweights[k] and multiply each by first, then sum
            //
            result[k] = tweights.data[k].iter().map(|&x| x * first).sum();
        }
        result
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

        // why isn't this loop reached?
        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];

            let agradnext = if i == self.layers.len() - 1 {
                activationgrad.clone()
            } else {
                self.layers[i + 1].activationgrad.clone()
            };

            let input = match i {
                0 => inputs.clone(),
                _ => self.layers[i - 1].activationdata.clone(),
            };


            let weightgrad = layer.weight_grad_backwards(&input, &agradnext);
            //update the activation gradient that is a trait of the layer
            //
            // ACTIVATION GRAD SHOULD NOT TAKE INPUTS
            self.layers[i].activationgrad = layer.activation_grad(&input, layer.weights.clone(), &agradnext);
            // update weights and biases
            //eprintln!("weightgrad: {:?}", &weightgrad.clone());
            self.layers[i].weights = self.layers[i].weights.clone() - (alpha * weightgrad);
            //self.layers[i].biases = subtract(&self.layers[i].biases, &biasgrad);
        }

        }

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
    fn network_test() {
        use crate::nn::Network;
        use crate::nn::Layer;
        use crate::activation::Activation;
        use crate::loss::*;

        let mut layer = Layer::new(2, 4, Activation::Relu);
        layer.biases = vec![0.0, 0.0, 0.0, 0.0];

        let mut layer2 = Layer::new(4, 2, Activation::Relu);
        layer2.biases = vec![0.0, 0.0];

        let mut network = Network::new();

        network.add_layer(layer.clone());
        network.add_layer(layer2.clone());

        let inputs = vec![1.0, 1.0];
        let target = vec![2.0, 2.0];

        eprintln!("initial network weights 0: {:?}", network.layers[0].weights);
        eprintln!("initial network weights 1: {:?}\n", network.layers[1].weights);

        let initial = network.forward(&inputs);
        network.forward(&inputs);

        for i in 0..100 {
            //eprintln!("network outputs on round {}: {:?}", i, network.forward(&inputs));
            network.forward(&inputs);
            network.backward(&inputs, &target, 0.004);
            eprintln!("loss on round {}: {:?}", i, network.loss.getloss(&network.layers[1].activationdata, &target));
            //eprintln!("network weights 0 updated: {:?}", network.layers[0].weights);
            //eprintln!("network weights 1 updated: {:?}", network.layers[1].weights);
        }

        eprintln!("initial network outputs: {:?}, \nfinal network outputs: {:?}", initial, network.forward(&inputs));

        assert_eq!(network.layers[0].weights.data[0][0], 1.1);

    }

}

