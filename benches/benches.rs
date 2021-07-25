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

pub fn cacheline_optimized_col_benchmark(c :&mut Criterion) {
    let a = Matrix::rand_matrix((100, 120));
    let b = Matrix::rand_matrix((120, 200));
    c.bench_function(
        "Cacheline Optimized Columns Multiplication",
        |bencher| bencher.iter(|| Matrix::cacheline_optimized_col_mul(&a, &b)));
}

pub fn preload_slice_benchmark(c: &mut Criterion) {
    let a = Matrix::rand_matrix((100, 120));
    let b = Matrix::rand_matrix((120, 200));
    c.bench_function(
        "Preload Slices Matrix Multiplication",
        |bencher| bencher.iter(|| Matrix::preload_slice_mul(&a, &b)));
}

criterion_group!(benches, naive_benchmark, cached_tdata_benchmark, cacheline_optimized_col_benchmark, preload_slice_benchmark);
criterion_main!(benches);
