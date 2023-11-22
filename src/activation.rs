pub mod Activation { 
    enum Activation {
        Relu(Relu),
        Tanh(Tanh),
        Sigmoid(Sigmoid),
        Softmax(Softmax),
    }

    struct Relu {}

    impl Relu {
        fn forward(&self, x: f64) -> f64 {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }

        fn backward(&self, x: f64, dy: f64) -> f64 {
            if x > 0.0 {
                dy
            } else {
                0.0
            
        }
    }

    struct Tanh {}

    impl Tanh {
        fn forward(&self, x: f64) -> f64 {
            2.0 / (1.0 + f64::exp(-2.0 * x)) - 1.0
        }

        fn backward(&self, x: f64, dy: f64) -> f64 {
            1.0 - self.forward(x).powi(2) * dy
        }
    }

    struct Sigmoid {}

    impl Sigmoid {
        fn forward(&self, x: f64) -> f64 {
            1.0 / (1.0 + f64::exp(-x))
        }

        fn backward(&self, x: f64, dy: f64) -> f64 {
            self.forward(x) * (1.0 - self.forward(x)) * dy
        }
    }

    struct Softmax {}

    impl Softmax {
        fn forward(&self, x: Vec<f64>) -> Vec<f64> {
            let mut output = Vec::new();
            let exp_sum = x.iter().map(|&xi| f64::exp(xi)).sum::<f64>();

            for &xi in &x {
                output.push(f64::exp(xi) / exp_sum);
            }

            output
        }

        fn backward(&self, x: Vec<f64>, dy: Vec<f64>) -> Vec<f64> {
            let mut gradient = Vec::new();

            for i in 0..x.len() {
                let xi = x[i];
                let dyi = dy[i];

                let mut sum = 0.0;
                for j in 0..x.len() {
                    let xj = x[j];
                    let djy = dy[j];

                    sum += xj * djy;
                }

                gradient.push(dyi - dyi * xi * sum);
            }

            gradient
        }
    }
}
