#[macro_use]
extern crate criterion;
extern crate fast_matrix_multiplication;
extern crate rand;

use criterion::{criterion_group, criterion_main, Criterion};
use fast_matrix_multiplication::Matrix;

pub fn naive_benchmark(c: &mut Criterion) {
    let a = rand_matrix((100, 120));
    let b = rand_matrix((120, 200));
    c.bench_function("Naive Multiplication", |bencher| bencher.iter(|| Matrix::naive_mul(&a, &b)));
}

fn rand_matrix(dim: (usize, usize)) -> Matrix {
    let mut mat = Matrix::new(dim);
    for i in 0..mat.dim.0 {
        for j in 0..mat.dim.1 {
            mat.data[i][j] = rand::random();
        }
    }
    mat
}

criterion_group!(benches, naive_benchmark);
criterion_main!(benches);
