extern crate criterion;
extern crate fmm;
extern crate rand;

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use fmm::Matrix;

fn bench_muls(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiplication");
    let a = Matrix::rand_matrix((100, 120));
    let b = Matrix::rand_matrix((120, 200));

    group.bench_function(
        "Naive Multiplication",
        |bencher| bencher.iter(|| Matrix::naive_mul(black_box(&a), black_box(&b))));
    group.bench_function(
        "Cached Matrix Transpose Multiplication",
        |bencher| bencher.iter(|| Matrix::cached_tdata_mul(black_box(&a), black_box(&b))));
    group.bench_function(
        "Cacheline Optimized Columns Multiplication",
        |bencher| bencher.iter(|| Matrix::cacheline_optimized_col_mul(black_box(&a), black_box(&b))));
}

criterion_group!(benches, bench_muls);
criterion_main!(benches);
