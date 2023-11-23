pub mod activations { 
    pub enum Activation {
        Relu,
        Tanh,
        Sigmoid,
        //Softmax,
    }

    trait ActivationFunction {
        fn forward(&self, x: f64) -> f64;
        fn backward(&self, x: f64, dy: f64) -> f64;
    }

    impl ActivationFunction for Activation { 
        fn forward(&self, x: f64) -> f64 {
            match self {
                Activation::Relu => if x > 0.0 { x } else { 0.0 },
                Activation::Tanh => 2.0 / (1.0 + f64::exp(-2.0 * x)) - 1.0,
                Activation::Sigmoid => 1.0 / (1.0 + f64::exp(-x)),
            }
        }
        fn backward(&self, x: f64, dy: f64) -> f64 {
            match self {
                Activation::Relu => if x > 0.0 { dy } else { 0.0 },
                Activation::Tanh => 1.0 - self.forward(x).powi(2) * dy,
                Activation::Sigmoid => self.forward(x) * (1.0 - self.forward(x)) * dy,
            }
        }
    }
}
