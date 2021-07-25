extern crate criterion;
extern crate fmm;
extern crate rand;

use criterion::{criterion_group, criterion_main, Criterion};
use fmm::Matrix;

pub fn naive_benchmark(c: &mut Criterion) {
    let a = Matrix::rand_matrix((100, 120));
    let b = Matrix::rand_matrix((120, 200));
    c.bench_function(
        "Naive Multiplication",
        |bencher| bencher.iter(|| Matrix::naive_mul(&a, &b)));
}

pub fn cached_tdata_benchmark(c: &mut Criterion) {
    let a = Matrix::rand_matrix((100, 120));
    let b = Matrix::rand_matrix((120, 200));
    c.bench_function(
        "Cached Matrix Transpose Multiplication",
        |bencher| bencher.iter(|| Matrix::cached_tdata_mul(&a, &b)));
}

fn rand_matrix(dim: (usize, usize)) -> Matrix {
    let mut mat = Matrix::new(dim);
    for i in 0..mat.dim.0 {
        for j in 0..mat.dim.1 {
            mat.set(i, j, rand::random());
        }
    }
    mat
}

criterion_group!(benches, naive_benchmark, cached_tdata_benchmark);
criterion_main!(benches);
