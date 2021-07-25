pub struct Matrix {
    pub data: Vec<Vec<i32>>,
    pub t_data: Vec<Vec<i32>>,
    pub dim: (usize, usize),
}

impl Matrix {
    pub fn new(dim: (usize, usize)) -> Self {
        let mut data = Vec::with_capacity(dim.0);
        for _ in 0..dim.0 {
            data.push(vec![0; dim.1]);
        }
        let mut t_data = Vec::with_capacity(dim.1);
        for _ in 0..dim.1 {
            t_data.push(vec![0; dim.0]);
        }
        Matrix {
            data,
            t_data,
            dim,
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
                let mut sum = 0;
                for (x, y) in row.iter().zip(b.col(j).iter()) {
                    sum += x * y;
                }
                mat.set(i, j, sum);
            }
        }
        mat
    }

    pub fn set(&mut self, i: usize, j: usize, val: i32) {
        self.data[i][j] = val;
        self.t_data[j][i] = val;
    }

    fn row(&self, idx: usize) -> &Vec<i32> {
        &self.data[idx]
    }

    fn col(&self, idx: usize) -> &Vec<i32> {
        &self.t_data[idx]
    }

    fn naive_col(&self, idx: usize) -> Vec<i32> {
        let mut vec = Vec::with_capacity(self.dim.0);
        for row in self.data.iter() {
            vec.push(row[idx]);
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
