pub mod activations {
    use crate::methods::loss::*;
    #[derive(Debug, PartialEq, Clone)]
    pub enum Activation {
        Relu,
        Tanh,
        Sigmoid,
        Softmax,
        None,
    }

    pub trait ActivationFunction {
        fn forward(&self, x: &[f64]) -> Vec<f64>;
        fn backward(&self, x: &[f64], dy: &[f64], l: &LossFunction) -> Vec<f64>;
    }

    impl ActivationFunction for Activation {
        fn forward(&self, x: &[f64]) -> Vec<f64> {
            match self {
                Activation::Relu => x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect(),
                Activation::Tanh => x.iter().map(|&v| 2.0 / (1.0 + f64::exp(-2.0 * v)) - 1.0).collect(),
                Activation::Sigmoid => x.iter().map(|&v| 1.0 / (1.0 + f64::exp(-v))).collect(),
                Activation::Softmax => softmax(x),
                Activation::None => x.to_vec(),
            }
        }

        fn backward(&self, x: &[f64], dy: &[f64], l: &LossFunction) -> Vec<f64> {
            match self {
                Activation::Relu => x.iter().zip(dy).map(|(&x, &dy)| if x > 0.0 { dy } else { 0.0 }).collect(),
                Activation::Tanh => x.iter().zip(dy).map(|(&x, &dy)| 1.0 - self.forward(&[x])[0].powi(2) * dy).collect(),
                Activation::Sigmoid => x.iter().zip(dy).map(|(&x, &dy)| {
                    let fwd = self.forward(&[x])[0];
                    fwd * (1.0 - fwd) * dy
                }).collect(),
                Activation::None => dy.to_vec(),
                Activation::Softmax => match l {
                    LossFunction::CrossEntropy => {
                        let y_pred = self.forward(x);
                        let mut y_true = vec![0.0; y_pred.len()];
                        y_true[dy[0] as usize] = 1.0;
                        y_pred.iter().zip(y_true).map(|(&y_pred, y_true)| y_pred - y_true).collect()
                    }
                    _ => panic!("Backward not implemented for Softmax"),
                }
            }
        }
    }

    fn softmax(values: &[f64]) -> Vec<f64> {
        // stability technique, subtract max value from each value
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = values.iter().map(|&v| f64::exp(v - max_val)).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&v| v / sum).collect()
    }
}

