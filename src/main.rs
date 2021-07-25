extern crate fmm;

use fmm::Matrix;

pub fn main() {
    let a = Matrix::rand_matrix((100, 120));
    let b = Matrix::rand_matrix((120, 200));
    let mul = Matrix::cached_tdata_mul(&a, &b);
    print!("{:?}", mul.dim);
}
