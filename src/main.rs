extern crate fmm;

use fmm::Matrix;

pub fn main() {
    let a = Matrix::rand_matrix((1000, 1000));
    let b = Matrix::rand_matrix((1000, 1000));
    let mul = Matrix::cacheline_optimized_col_mul(&a, &b);
    print!("{:?}", mul.dim);
}
