use std::ops::Mul;
use std::ops::Add;

#[derive(Debug, PartialEq)]
struct Matrix {
    nrows: usize,
    ncols: usize,
    data: Vec<Vec<f64>>,
}

#[derive(Debug, PartialEq)]
struct Vector {
    data: Vec<f64>,
}


impl Matrix { 
    fn nrows(&self) -> usize {
        self.data.len()
    }
    fn ncols(&self) -> usize {
        self.data[0].len()
    }

    fn new(nrows: usize, ncols: usize) -> Self {
        Matrix {
            nrows,
            ncols,
            data: vec![vec![0.0; ncols]; nrows],
        }
    }

}

impl Vector { 
    fn len(&self) -> usize {
        self.data.len()
    }

    fn new(len: usize) -> Self {
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

#[derive(Debug, PartialEq)]
struct Layer {
    weights: Matrix,
    biases: Vector,
}

impl Layer { 
    fn forward(&self, inputs: &Vector) -> Vector {
        self.weights.clone() * inputs.clone() + self.biases.clone()
    }
}

fn main() { 
    println!("Hello World");
}


#[cfg(test)]
mod tests {
    use super::*;

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
        let layer = Layer { 
            weights: Matrix { 
                nrows: 2,
                ncols: 2,
                data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            },
            biases: Vector { 
                data: vec![0.0, 0.0],
            },
        };

        let inputs = Vector { data: vec![1.0, 2.0] };
        let outputs = Vector { data: vec![5.0, 11.0] };
        let layer_outputs = layer.forward(&inputs);
        assert_eq!(layer_outputs.clone(), outputs.clone());
        let layer2 = layer.forward(&layer_outputs);
        assert_eq!(layer2, Vector { data: vec![27.0, 59.0] });
    }
}

