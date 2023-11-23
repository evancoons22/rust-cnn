pub mod activations { 
    #[derive(Debug, PartialEq)]
    pub enum Activation {
        Relu,
        Tanh,
        Sigmoid,
        None,
        //Softmax,
    }

    impl Clone for Activation {
        fn clone(&self) -> Self {
            match self {
                Activation::Relu => Activation::Relu,
                Activation::Tanh => Activation::Tanh,
                Activation::Sigmoid => Activation::Sigmoid,
                Activation::None => Activation::None,
            }
        }
    }

    pub trait ActivationFunction {
        fn forward(&self, x: f64) -> f64;
        fn backward(&self, x: f64, dy: f64) -> f64;
    }

    // implement ActivatonFunction for Activation
    impl ActivationFunction for Activation { 
        fn forward(&self, x: f64) -> f64 {
            match self {
                Activation::Relu => if x > 0.0 { x } else { 0.0 },
                Activation::Tanh => 2.0 / (1.0 + f64::exp(-2.0 * x)) - 1.0,
                Activation::Sigmoid => 1.0 / (1.0 + f64::exp(-x)),
                Activation::None => x
            }
        }
        fn backward(&self, x: f64, dy: f64) -> f64 {
            match self {
                Activation::Relu => if x > 0.0 { dy } else { 0.0 },
                Activation::Tanh => 1.0 - self.forward(x).powi(2) * dy,
                Activation::Sigmoid => self.forward(x) * (1.0 - self.forward(x)) * dy,
                Activation::None => dy,
            }
        }
    }
}
