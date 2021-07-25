extern crate rand;

const LINESIZE: usize = 64;

pub struct Matrix {
    pub data: Vec<i32>,
    pub tdata: Vec<i32>,
    pub dim: (usize, usize),
    row_size: usize,
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
        // One extra row is allocated for preloading in preload_slice_mul
        let tdata = vec![0; tdata_row_size * (dim.1 + 1)];
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

    pub fn preload_slice_mul(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.dim.1, b.dim.0);
        let mut mat = Matrix::new((a.dim.0, b.dim.1));
        let mut row = a.row(0);
        for i in 0..a.dim.0 {
            let mut col = b.col(0);
            for j in 0..b.dim.1 {
                let sum = row.iter().zip(col.iter())
                    .map(|(x, y)| x * y).sum();
                mat.set(i, j, sum);
                col = b.col(j + 1);
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

    pub fn rand_matrix(dim: (usize, usize)) -> Matrix {
        let mut mat = Matrix::new(dim);
        for i in 0..mat.dim.0 {
            for j in 0..mat.dim.1 {
                mat.set(i, j, rand::random::<i16>() as i32);
            }
        }
        mat
    }
}
