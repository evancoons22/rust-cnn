use std::ops::Mul;
use std::ops::Index;

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
                .map(|_| (0..ncols).map(|_| rng.gen_range(-0.5..0.5)).collect())
                .collect(),
        }
    }

    //implement copy trait

}

//implement indexing for matrix
impl Index<usize> for Matrix {
    type Output = Vec<f64>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl Mul<f64> for Matrix {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        let mut result = Matrix::new(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                result.data[i][j] = self.data[i][j] * rhs;
            }
        }
        result
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

pub fn scalar_mul_vec(scalar: f64, vec: &[f64]) -> Vec<f64> {
    let new = vec.iter().map(|&x| scalar * x).collect();
    return new
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


#[cfg(test)]
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
fn rand_matrix() {
    let matrix = Matrix::rand(2, 2);
    assert_eq!(matrix.nrows(), 2);
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
fn test_dot_product() {
    use crate::linalg::dot_product;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let c: f64 = 14.0;
    assert_eq!(dot_product(&a, &b), c);
}
