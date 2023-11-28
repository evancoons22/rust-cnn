pub mod linalg {

    use std::ops::Mul;
    
    #[derive(Debug, PartialEq)]
    pub struct Matrix {
        pub nrows: usize,
        pub ncols: usize,
        pub data: Vec<Vec<f64>>,
    }

    use rand::Rng;

    impl Matrix { 
        pub fn nrows(&self) -> usize {
            self.data.len()
        }
        pub fn ncols(&self) -> usize {
            self.data[0].len()
        }
        pub fn new(nrows: usize, ncols: usize) -> Self {
            Matrix {
                nrows,
                ncols,
                data: vec![vec![0.0; ncols]; nrows],
            }
        }
        pub fn rand(nrows: usize, ncols: usize) -> Self {
            let mut rng = rand::thread_rng();
            Matrix {
                nrows,
                ncols,
                data: (0..nrows)
                    .map(|_| (0..ncols).map(|_| rng.gen_range(-1.0..1.0)).collect())
                    .collect(),
            }
        }

    }


    impl Mul<Matrix> for Matrix {
        type Output = Self;

        fn mul(self, rhs: Matrix) -> Self {
            assert_eq!(self.ncols(), rhs.nrows());
            let mut result = Matrix::new(self.nrows(), rhs.ncols());
            for i in 0..self.nrows() {
                for j in 0..rhs.ncols() {
                    for k in 0..self.ncols() {
                        result.data[i][j] += self.data[i][k] * rhs.data[k][j];
                    }
                }
            }
            result
        }
    }

    impl Mul<Vec<f64>> for Matrix {
        type Output = Vec<f64>;

        fn mul(self, rhs: Vec<f64>) -> Vec<f64> {
            assert_eq!(self.ncols(), rhs.len());
            //let mut result = Vec<f64>::new(self.nrows());
            let mut result = vec![0.0; self.nrows()];
            for i in 0..self.nrows() {
                for j in 0..rhs.len() {
                    result[i] += self.data[i][j] * rhs[j];
                }
            }
            result
        }
    }

    impl Mul<&Vec<f64>> for &Matrix {
        type Output = Vec<f64>;
        fn mul(self, rhs: &Vec<f64>) -> Vec<f64> {
            assert_eq!(self.ncols(), rhs.len());
            //let mut result = Vec<f64>::new(self.nrows());
            let mut result = vec![0.0; self.nrows()];
            for i in 0..self.nrows() {
                for j in 0..rhs.len() {
                    result[i] += self.data[i][j] * rhs[j];
                }
            }
            result
        }
    }

    // implement subtract elementwise for matrices and vectors
    use std::ops::{Sub, Add};
    impl Sub<Matrix> for Matrix {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            assert_eq!(self.nrows(), rhs.nrows());
            assert_eq!(self.ncols(), rhs.ncols());
            let mut result = Matrix::new(self.nrows(), self.ncols());
            for i in 0..self.nrows() {
                for j in 0..self.ncols() {
                    result.data[i][j] = self.data[i][j] - rhs.data[i][j];
                }
            }
            result
        }
    }

    impl Add<Matrix> for Matrix {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            assert_eq!(self.nrows(), rhs.nrows());
            assert_eq!(self.ncols(), rhs.ncols());
            let mut result = Matrix::new(self.nrows(), self.ncols());
            for i in 0..self.nrows() {
                for j in 0..self.ncols() {
                    result.data[i][j] = self.data[i][j] + rhs.data[i][j];
                }
            }
            result
        }
    }

    impl Mul<Matrix> for f64 {
        type Output = Matrix;
        fn mul(self, rhs: Matrix) -> Matrix {
            let mut result = Matrix::new(rhs.nrows(), rhs.ncols());
            for i in 0..rhs.nrows() {
                for j in 0..rhs.ncols() {
                    result.data[i][j] = self * rhs.data[i][j];
                }
            }
            result
        }
    }


    pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() {
            panic!("Vectors must be of equal length");
        }
        vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x * y).sum()
    }

    pub fn transpose(matrix: Matrix) -> Matrix {
        let mut result = Matrix::new(matrix.ncols(), matrix.nrows());
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                result.data[j][i] = matrix.data[i][j];
            }
        }
        result
    }

    pub fn add(vec1: &[f64], vec2: &[f64]) -> Vec<f64> {
        if vec1.len() != vec2.len() {
            panic!("Vectors must be of equal length");
        }
        vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x + y).collect()
    }

    pub fn subtract(vec1: &[f64], vec2: &[f64]) -> Vec<f64> { 
        if vec1.len() != vec2.len() { 
            panic!("Vectors must be of equal length");
        }
        vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x - y).collect()
    }

    impl Clone for Matrix {
        fn clone(&self) -> Self {
            Matrix { data: self.data.clone(), nrows: self.nrows, ncols: self.ncols }
        }
    }

}

pub mod nn { 
    use super::linalg::Matrix;
    use super::linalg::add;
    use crate::activation::activations::Activation;
    use crate::activation::activations::ActivationFunction;
    use super::loss::*;
    use super::linalg::subtract;

    #[derive(Debug, PartialEq)]
    pub struct Layer {
        pub input_size: usize,
        pub output_size: usize,
        pub weights: Matrix,
        pub biases: Vec<f64>,
        pub activation: Activation,
        pub activationdata: Vec<f64>,
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
            }
        }
    }

    impl Layer { 
        pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
            // applies network weights AND activates
            self.activationdata = self.activation.forward(&add(&(&self.weights * &inputs), &self.biases));
            self.activationdata.clone()
        }
        pub fn new(input_size: usize, output_size: usize) -> Self {
            Layer {
                input_size,
                output_size,
                weights: Matrix::rand(output_size, input_size),
                biases: vec![0.0; output_size],
                activation: Activation::None,
                activationdata: vec![0.0; output_size],
            }
        }

        pub fn weight_grad_backwards(&self, inputs: &Vec<f64>, a: &Vec<f64>, agradnext: &Vec<f64>) -> Matrix {
            let mut result = Matrix::new(self.output_size, self.input_size);
            let actgrad = self.activation.backward(&inputs, &LossFunction::MSE);
            for j in 0..self.output_size {
                for k in 0..self.input_size {
                    result.data[k][j] = actgrad[j] * a[k] * agradnext[j];
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


        pub fn activation_grad_backward(&self, inputs: &Vec<f64>, weights: Matrix, agradnext: &Vec<f64>) -> Vec<f64> {
            use super::linalg::*;
            let mut result = vec![0.0; self.output_size];
            let actgrad = self.activation.backward(&inputs, &LossFunction::MSE);
            let tweights = transpose(weights.clone());
            for k in 0..self.input_size {
                result[k] += dot_product(&tweights.data[k], &add(&actgrad, &agradnext));
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

        pub fn backward(&mut self, inputs: &Vec<f64>, outputs: &Vec<f64>, y_true: &Vec<f64>, alpha: f64) {
            let mut prevactgrad = self.loss.backward(&outputs, &y_true);
            let mut prevweights = self.layers.last().unwrap().weights.clone();
            let mut allweightupdates: Vec<Matrix> = Vec::new();
            let mut allbiasupdates: Vec<Vec<f64>> = Vec::new();
            for layer in self.layers.iter().rev() {
                let weightgrad = layer.weight_grad_backwards(&inputs, &layer.activationdata, &prevactgrad);
                allweightupdates.push(weightgrad);
                let biasgrad = layer.bias_grad_backwards(&inputs, &prevactgrad);
                allbiasupdates.push(biasgrad);
                let actgrad = layer.activation_grad_backward(&inputs, prevweights, &prevactgrad);
                prevweights = layer.weights.clone();
                prevactgrad = actgrad;

            }
            allweightupdates.reverse();
            for (i, layer) in self.layers.iter_mut().enumerate() {
                layer.weights = layer.weights.clone() - alpha * allweightupdates[i].clone();
                layer.biases = subtract(&layer.biases.clone(), &allbiasupdates[i].clone());
            }
        }

    }
}

pub mod loss {

    use super::linalg::{subtract, dot_product};

    #[derive(Debug, PartialEq)]
    pub enum LossFunction {
        MSE,
        CrossEntropy,
    }

    pub trait Loss {
        fn getloss(&self, y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64;
        fn backward(&self, y_pred: &Vec<f64>, y_true: &Vec<f64>) -> Vec<f64>;
    }

    impl Clone for LossFunction {
        fn clone(&self) -> Self {
            match self {
                LossFunction::MSE => LossFunction::MSE,
                LossFunction::CrossEntropy => LossFunction::CrossEntropy,
            }
        }
    }

    impl Loss for LossFunction{
        fn getloss(&self, y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
            match &self {
                LossFunction::MSE => (y_pred.len() as f64) * dot_product(&subtract(y_pred, y_true), &subtract(y_pred, y_true)),
                LossFunction::CrossEntropy => y_pred.iter().zip(y_true.iter()).map(|(x, y)| -1.0 * (y * x.ln())).sum::<f64>(),
            }
        }

        fn backward(&self, y_pred: &Vec<f64>, y_true: &Vec<f64>) -> Vec<f64> {
            match self {
                LossFunction::MSE => y_pred.iter().zip(y_true.iter()).map(|(x, y)| 2.0 * (x - y)).collect::<Vec<f64>>(),
                LossFunction::CrossEntropy => y_pred.iter().zip(y_true.iter()).map(|(x, y)| x - y).collect::<Vec<f64>>(),
            }

        }
    }


}

#[cfg(test)]
mod tests {
    use super::linalg::{Matrix, dot_product};
    use super::nn::{Layer, Network};

    #[test]
    fn test_mat_mat_mul() {
        let a = Matrix {
            nrows: 2,
            ncols: 3,
            data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        };
        let b = Matrix {
            nrows: 3,
            ncols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        };
        let c = Matrix {
            nrows: 2,
            ncols: 2,
            data: vec![vec![22.0, 28.0], vec![49.0, 64.0]],
        };
        assert_eq!(a * b, c);
    }
    #[test]
    fn test_mat_vec_mul() {
        let a = Matrix {
            nrows: 2,
            ncols: 3,
            data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        };
        let b = vec![1.0, 2.0, 3.0];
        let c =  vec![14.0, 32.0];
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_vec_vec_mul() { 
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c: f64 = 14.0;
        assert_eq!(dot_product(&a, &b), c);
    }

    #[test]
    fn test_layer_forward() { 
        use crate::activation::activations::Activation;
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
        use crate::activation::activations::Activation;
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
        };

        let inputs = vec![1.0, 2.0];
        let mut network = Network { layers: vec![layer.clone(), layer.clone()], loss: super::loss::LossFunction::MSE };
        assert_eq!(network.forward(&inputs), vec![27.0, 59.0]);

        }

    #[test]
    fn layer_forward_new() { 
        let mut layer = Layer::new(2, 2);
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
    fn rand_matrix() {
        let matrix = Matrix::rand(2, 2);
        assert_eq!(matrix.nrows(), 2);
    }

    #[test]
    fn activation_test() { 
        let mut layer = Layer::new(2, 2);
        layer.weights = Matrix { 
            nrows: 2,
            ncols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        layer.activation = crate::activation::activations::Activation::Sigmoid;
        let inputs = vec![1.0, 2.0];
        let outputs = layer.forward(&inputs);
        println!("{:?}", outputs);
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_loss() {
        use super::loss::LossFunction;
        use super::loss::Loss;
        let y_pred = vec![0.0, 0.0, 1.0];
        let y_true = vec![0.0, 0.0, 1.0];
        let loss = LossFunction::MSE;
        assert_eq!(loss.getloss(&y_pred, &y_true), 0.0);
    }

    #[test]
    fn test_loss_backward() {
        use super::loss::LossFunction;
        use super::loss::Loss;
        let y_pred = vec![0.0, 0.0, 1.0];
        let y_true = vec![0.0, 0.0, 1.0];
        let loss = LossFunction::MSE;
        assert_eq!(loss.backward(&y_pred, &y_true), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c: f64 = 14.0;
        assert_eq!(super::linalg::dot_product(&a, &b), c);
    }

    #[test]
    fn test_loss_cross_entropy() {
        use super::loss::LossFunction;
        use super::loss::Loss;
        let y_pred = vec![0.001, 0.001, 0.998];
        let y_true = vec![0.0, 0.0, 1.0];
        let loss = LossFunction::CrossEntropy;
        assert_eq!(loss.getloss(&y_pred, &y_true), 0.0020020026706730793);
        
    }
    #[test]
    fn test_elementwise_matrix_add() {
        let a = Matrix { 
            nrows: 2,
            ncols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        let b = Matrix { 
            nrows: 2,
            ncols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        let c = Matrix { 
            nrows: 2,
            ncols: 2,
            data: vec![vec![2.0, 4.0], vec![6.0, 8.0]],
        };
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_loss_cross_entropy_backward() {
        use super::loss::LossFunction;
        use super::loss::Loss;
        let y_pred = vec![0.001, 0.001, 0.998];
        let y_true = vec![0.0, 0.0, 1.0];
        let loss = LossFunction::CrossEntropy;
        assert_eq!(loss.backward(&y_pred, &y_true), vec![0.001,0.001, -0.002]);
        
    }

    #[test]
    fn update_network() {
        use super::nn::Network;
        use super::nn::Layer;
        let mut layer = Layer::new(2, 2);
        layer.weights = Matrix { 
            nrows: 2,
            ncols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        layer.biases = vec![0.0, 0.0];
        let mut network = Network::new();
        network.add_layer(layer.clone());
        eprintln!("{:?}", network.layers[0].biases);
        let inputs = vec![1.0, 2.0];
        //let outputs = vec![5.0, 11.0];
        let y_true = vec![0.0, 0.0];
        let alpha = 0.01;
        let layer_outputs = layer.forward(&inputs);
        //assert_eq!(layer_outputs.clone(), outputs.clone());
        let layer2 = layer.forward(&layer_outputs);
        //assert_eq!(layer2, vec![27.0, 59.0]);
        network.backward(&inputs, &layer2, &y_true, alpha);
        eprintln!("{:?}", network.layers[0].biases);
        assert_eq!(network.layers[0].weights.data[0][0], 0.9999999999999999);
    }   

}

