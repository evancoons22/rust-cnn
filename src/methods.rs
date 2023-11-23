pub mod linalg {

    use std::ops::Mul;
    use std::ops::Add;
    
    #[derive(Debug, PartialEq)]
    pub struct Matrix {
        pub nrows: usize,
        pub ncols: usize,
        pub data: Vec<Vec<f64>>,
    }

    #[derive(Debug, PartialEq)]
    pub struct Vector {
        pub data: Vec<f64>,
    }


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
            use rand::Rng;
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

    impl Vector { 
        pub fn len(&self) -> usize {
            self.data.len()
        }

        pub fn new(len: usize) -> Self {
            Vector {
                data: vec![0.0; len],
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

    impl Mul<Vector> for Matrix {
        type Output = Vector;

        fn mul(self, rhs: Vector) -> Vector {
            assert_eq!(self.ncols(), rhs.len());
            let mut result = Vector::new(self.nrows());
            for i in 0..self.nrows() {
                for j in 0..rhs.len() {
                    result.data[i] += self.data[i][j] * rhs.data[j];
                }
            }
            result
        }
    }

    impl Mul<Vector> for Vector {
        type Output = f64;

        fn mul(self, rhs: Vector) -> f64 {
            assert_eq!(self.len(), rhs.len());
            let mut result = 0.0;
            for i in 0..self.len() {
                result += self.data[i] * rhs.data[i];
            }
            result
        }
    }

    impl Add<Vector> for Vector {
        type Output = Self;

        fn add(self, rhs: Vector) -> Self {
            assert_eq!(self.len(), rhs.len());
            let mut result = Vector::new(self.len());
            for i in 0..self.len() {
                result.data[i] = self.data[i] + rhs.data[i];
            }
            result
        }
    }

    impl Clone for Vector {
        fn clone(&self) -> Self {
            Vector { data: self.data.clone() }
        }
    }

    impl Clone for Matrix {
        fn clone(&self) -> Self {
            Matrix { data: self.data.clone(), nrows: self.nrows, ncols: self.ncols }
        }
    }

}

pub mod nn { 
    use super::linalg::{Matrix, Vector};
    use crate::activation::activations::Activation;

    #[derive(Debug, PartialEq)]
    pub struct Layer {
        pub input_size: usize,
        pub output_size: usize,
        pub weights: Matrix,
        pub biases: Vector,
        pub activation: Activation,
    }

    impl Clone for Layer {
        fn clone(&self) -> Self {
            Layer { 
                input_size: self.input_size,
                output_size: self.output_size,
                weights: self.weights.clone(),
                biases: self.biases.clone(),
                activation: self.activation.clone(),
            }
        }
    }

    impl Layer { 
        pub fn forward(&self, inputs: &Vector) -> Vector {
            self.weights.clone() * inputs.clone() + self.biases.clone()
        }
        pub fn new(input_size: usize, output_size: usize) -> Self {
            Layer {
                input_size,
                output_size,
                weights: Matrix::rand(output_size, input_size),
                biases: Vector::new(output_size),
                activation: Activation::None,
            }
        }
    }

    pub struct Network {
        pub layers: Vec<Layer>,
    }

    impl Network { 
        pub fn new() -> Self {
            Network {
                layers: Vec::new(),
            }
        }

        pub fn add_layer(&mut self, layer: Layer) {
            self.layers.push(layer);
        }

        pub fn forward(&self, inputs: &Vector) -> Vector {
            let mut outputs = inputs.clone();
            for layer in &self.layers {
                outputs = layer.forward(&outputs);
            }
            outputs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::linalg::{Matrix, Vector};
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
        let b = Vector {
            data: vec![1.0, 2.0, 3.0],
        };
        let c = Vector {
            data: vec![14.0, 32.0],
        };
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_vec_vec_mul() { 
        let a = Vector { data: vec![1.0, 2.0, 3.0] };
        let b = Vector { data: vec![1.0, 2.0, 3.0] };
        let c: f64 = 14.0;
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_layer_forward() { 
        use crate::activation::activations::Activation;
        let layer = Layer { 
            input_size: 2,
            output_size: 2,
            weights: Matrix { 
                nrows: 2,
                ncols: 2,
                data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            },
            biases: Vector { 
                data: vec![0.0, 0.0],
            },
            activation: Activation::None,
        };

        let inputs = Vector { data: vec![1.0, 2.0] };
        let outputs = Vector { data: vec![5.0, 11.0] };
        let layer_outputs = layer.forward(&inputs);
        assert_eq!(layer_outputs.clone(), outputs.clone());
        let layer2 = layer.forward(&layer_outputs);
        assert_eq!(layer2, Vector { data: vec![27.0, 59.0] });
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
            biases: Vector { 
                data: vec![0.0, 0.0],
            },
            activation: Activation::None,
        };

        let inputs = Vector { data: vec![1.0, 2.0] };
        let network = Network { layers: vec![layer.clone(), layer.clone()] };
        assert_eq!(network.forward(&inputs), Vector { data: vec![27.0, 59.0] })

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
        let inputs = Vector { data: vec![1.0, 2.0] };
        let outputs = layer.forward(&inputs);
        assert_eq!(outputs.len(), 2);
    }

    #[test] 
    fn rand_matrix() {
        let matrix = Matrix::rand(2, 2);
        assert_eq!(matrix.nrows(), 2);
    }
}

