use crate::linalg::*;

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

impl Loss for LossFunction {
    fn getloss(&self, y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
        match &self {
            LossFunction::MSE => {
                (y_pred.len() as f64)
                    * dot_product(&subtract(y_pred, y_true), &subtract(y_pred, y_true))
            }
            LossFunction::CrossEntropy => y_pred
                .iter()
                .zip(y_true.iter())
                .map(|(x, y)| -1.0 * (y * x.ln()))
                .sum::<f64>(),
        }
    }

    fn backward(&self, y_pred: &Vec<f64>, y_true: &Vec<f64>) -> Vec<f64> {
        match self {
            LossFunction::MSE => y_pred
                .iter()
                .zip(y_true.iter())
                .map(|(x, y)| 2.0 * (x - y))
                .collect::<Vec<f64>>(),
            LossFunction::CrossEntropy => y_pred
                .iter()
                .zip(y_true.iter())
                .map(|(x, y)| x - y)
                .collect::<Vec<f64>>(),
        }
    }
}

#[cfg(test)]
#[test]
fn test_loss_cross_entropy_backward() {
    use crate::loss::Loss;
    use crate::loss::LossFunction;
    let y_pred = vec![0.001, 0.001, 0.998];
    let y_true = vec![0.0, 0.0, 1.0];
    let loss = LossFunction::CrossEntropy;
    assert_eq!(loss.backward(&y_pred, &y_true), vec![0.001, 0.001, -0.002]);
}

#[test]
fn test_loss_cross_entropy() {
    use crate::loss::Loss;
    use crate::loss::LossFunction;
    let y_pred = vec![0.001, 0.001, 0.998];
    let y_true = vec![0.0, 0.0, 1.0];
    let loss = LossFunction::CrossEntropy;
    assert_eq!(loss.getloss(&y_pred, &y_true), 0.0020020026706730793);
}

#[test]
fn test_loss_mse() {
    use crate::loss::Loss;
    use crate::loss::LossFunction;
    let y_pred = vec![0.0, 0.0, 1.0];
    let y_true = vec![0.0, 0.0, 1.0];
    let loss = LossFunction::MSE;
    assert_eq!(loss.getloss(&y_pred, &y_true), 0.0);
}

#[test]
fn test_loss_mse_backward() {
    use crate::loss::Loss;
    use crate::loss::LossFunction;
    let y_pred = vec![0.0, 0.0, 1.0];
    let y_true = vec![0.0, 0.0, 1.0];
    let loss = LossFunction::MSE;
    assert_eq!(loss.backward(&y_pred, &y_true), vec![0.0, 0.0, 0.0]);
}
