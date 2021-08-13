extern crate rand;

const LINESIZE: usize = 64;
const INTS_PER_LINE: usize = LINESIZE / 4;

pub struct Matrix {
    pub data: Vec<i32>,
    pub dim: (usize, usize),
    row_size: usize,
    pub tdata: Vec<i32>,
    tdata_row_size: usize,
}

impl Matrix {
    pub fn new(dim: (usize, usize)) -> Self {
        // Allocate Vec for data, aligning each row with cache line
        let line_per_row = (dim.1 * 4) / LINESIZE +
            ((dim.1 % LINESIZE != 0) as usize);
        let row_size = line_per_row * LINESIZE / 4;
        // One extra row is allocated for preloading in preload_slice_mul
        let data = vec![0; row_size * (dim.0 + 1)];
        // Allocate Vec for tdata, aligning each row with cache line
        let line_per_row = (dim.0 * 4) / LINESIZE +
            ((dim.0 % LINESIZE != 0) as usize);
        let tdata_row_size = line_per_row * LINESIZE / 4;
        let tdata = vec![0; tdata_row_size * dim.1];
        Matrix {
            data,
            tdata,
            dim,
            row_size,
            tdata_row_size,
        }
    }

    pub fn naive_mul(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.dim.1, b.dim.0);
        let mut mat = Matrix::new((a.dim.0, b.dim.1));
        for i in 0..a.dim.0 {
            let row = a.row(i);
            for j in 0..b.dim.1 {
                let mut sum = 0;
                for (x, y) in row.iter().zip(b.naive_col(j).iter()) {
                    sum += x * y;
                }
                mat.set(i, j, sum);
            }
        }
        mat
    }

    pub fn cached_tdata_mul(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.dim.1, b.dim.0);
        let mut mat = Matrix::new((a.dim.0, b.dim.1));
        for i in 0..a.dim.0 {
            let row = a.row(i);
            for j in 0..b.dim.1 {
                let sum = row.iter().zip(b.col(j).iter())
                    .map(|(x, y)| x * y).sum();
                mat.set(i, j, sum);
            }
        }
        mat
    }

    pub fn cacheline_optimized_col_mul(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.dim.1, b.dim.0);
        let mut mat = Matrix::new((a.dim.0, b.dim.1));
        let mut buffer = [0i32; INTS_PER_LINE];
        for i in 0..mat.dim.0 {
            let row = a.row(i);
            for j in (0..mat.dim.1).step_by(INTS_PER_LINE) {
                let cols = b.cols_iter(j);
                row.iter().zip(cols)
                    .for_each(|(row_ele, cols_ele)|
                        Matrix::vec_scalar_mul(row_ele, cols_ele, &mut buffer));
                let start = i * mat.row_size + j;
                mat.data[start..start + INTS_PER_LINE].copy_from_slice(&buffer[..]);
                buffer = [0i32; INTS_PER_LINE];
            }
        }
        mat
    }

    /// Perform scalar multiplication on vec, then adds the result to output
    /// using SIMD instructions
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),
                  target_feature = "avx2"))]
    fn vec_scalar_mul(
        scalar: &i32,
        vec: &[i32],
        output: &mut [i32],
    ) {
        unsafe {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::{
                _mm256_set_epi32, _mm256_load_si256, _mm256_mullo_epi32,
                _mm256_add_epi32, __m256i};
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::{
                _mm256_set_epi32, _mm256_load_si256, _mm256_mullo_epi32,
                _mm256_add_epi32, __m256i};
            let scalar = _mm256_set_epi32(
                scalar.clone(), scalar.clone(), scalar.clone(), scalar.clone(),
                scalar.clone(), scalar.clone(), scalar.clone(), scalar.clone());
            // Multiplication and addition on first 8 ints
            let simd_vec = _mm256_load_si256(vec.as_ptr() as *const __m256i);
            let simd_vec = _mm256_mullo_epi32(scalar, simd_vec);
            let buffer = _mm256_load_si256(output.as_ptr() as *const __m256i);
            let buffer = _mm256_add_epi32(simd_vec, buffer);
            output[0..8].copy_from_slice(
                &std::mem::transmute::<__m256i, [i32; 8]>(buffer));
            // Multiplication and addition on second 8 ints
            let simd_vec = _mm256_load_si256(vec.as_ptr().add(8) as *const __m256i);
            let simd_vec = _mm256_mullo_epi32(scalar, simd_vec);
            let buffer = _mm256_load_si256(output.as_ptr().add(8) as *const __m256i);
            let buffer = _mm256_add_epi32(simd_vec, buffer);
            output[8..INTS_PER_LINE].copy_from_slice(
                &std::mem::transmute::<__m256i, [i32; 8]>(buffer));
        }
    }

    /// Perform scalar multiplication on vec, then adds the result to output
    #[cfg(not(target_feature = "avx2"))]
    fn vec_scalar_mul(
        scalar: &i32,
        vec: &[i32],
        output: &mut [i32],
    ) {
        for (i, col_ele) in vec.iter().enumerate() {
            output[i] += scalar * col_ele;
        }
    }

    pub fn preload_slice_mul(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.dim.1, b.dim.0);
        let mut mat = Matrix::new((a.dim.0, b.dim.1));
        let mut buffer = [0i32; INTS_PER_LINE];
        let mut row = a.row(0);
        for i in 0..mat.dim.0 {
            let mut cols = b.cols_iter(0);
            for j in (0..mat.dim.1).step_by(INTS_PER_LINE) {
                row.iter().zip(cols)
                    .for_each(|(row_ele, cols_ele)|
                        Matrix::vec_scalar_mul(row_ele, cols_ele, &mut buffer));
                let start = i * mat.row_size + j;
                mat.data[start..start + INTS_PER_LINE].copy_from_slice(&buffer[..]);
                buffer = [0i32; INTS_PER_LINE];
                cols = b.cols_iter(j + INTS_PER_LINE);
            }
            row = a.row(i + 1);
        }
        mat
    }

    pub fn get(&self, i: usize, j: usize) -> i32 {
        return self.data[i * self.row_size + j];
    }

    pub fn set(&mut self, i: usize, j: usize, val: i32) {
        self.data[i * self.row_size + j] = val;
        self.tdata[j * self.tdata_row_size + i] = val;
    }

    fn row(&self, idx: usize) -> &[i32] {
        let start = idx * self.row_size;
        &self.data[start..start + self.dim.1]
    }

    fn col(&self, idx: usize) -> &[i32] {
        let start = idx *self.tdata_row_size;
        &self.tdata[start..start + self.dim.0]
    }

    fn naive_col(&self, idx: usize) -> Vec<i32> {
        let mut vec = Vec::with_capacity(self.dim.0);
        for row in 0..self.dim.0 {
            vec.push(self.get(row, idx));
        }
        vec
    }

    fn cols_iter(&self, idx: usize) -> ColsIter {
        ColsIter::new(self, idx)
    }

    pub fn rand_matrix(dim: (usize, usize)) -> Matrix {
        let mut mat = Matrix::new(dim);
        for i in 0..mat.dim.0 {
            for j in 0..mat.dim.1 {
                mat.set(i, j, rand::random::<i8>() as i32);
            }
        }
        mat
    }
}

/// Iterate through elements of INTS_PER_LINE columns at once
struct ColsIter<'a> {
    mat: &'a Matrix,
    col_idx: usize,
    idx: usize,
}

impl<'a> ColsIter<'a> {
    pub fn new(mat: &'a Matrix, col_idx: usize) -> Self {
        ColsIter {
            mat,
            col_idx,
            idx: 0,
        }
    }
}

impl<'a> std::iter::Iterator for ColsIter<'a> {
    type Item = &'a [i32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.mat.dim.0 {
            None
        } else {
            let data_idx = self.idx * self.mat.row_size + self.col_idx;
            self.idx += 1;
            Some(&self.mat.data[data_idx..data_idx + INTS_PER_LINE])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplications() {
        let a = Matrix::rand_matrix((24, 24));
        let b = Matrix::rand_matrix((24, 24));
        let naive = Matrix::naive_mul(&a, &b);
        assert_eq!(naive.data, Matrix::cached_tdata_mul(&a, &b).data);
        assert_eq!(naive.data, Matrix::cacheline_optimized_col_mul(&a, &b).data);
        assert_eq!(naive.data, Matrix::preload_slice_mul(&a, &b).data);
    }

    #[test]
    fn test_vec_scalar_mul() {
        let vec = [2i32; INTS_PER_LINE];
        let mut buffer  = [0i32; INTS_PER_LINE];
        Matrix::vec_scalar_mul(&5, &vec[..], &mut buffer);
        assert_eq!([10; INTS_PER_LINE], buffer);
    }
}
