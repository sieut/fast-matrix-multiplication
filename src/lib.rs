pub struct Matrix {
    pub data: Vec<Vec<i32>>,
    pub dim: (usize, usize),
}

impl Matrix {
    pub fn new(dim: (usize, usize)) -> Self {
        let mut data = Vec::with_capacity(dim.0);
        for _ in 0..dim.1 {
            data.push(vec![0; dim.1]);
        }
        Matrix {
            data,
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
                for (x, y) in row.iter().zip(b.col(j).iter()) {
                    sum += x * y;
                }
                mat.data[i][j] = sum;
            }
        }
        mat
    }

    fn row(&self, idx: usize) -> Vec<i32> {
        self.data[idx].clone()
    }

    fn col(&self, idx: usize) -> Vec<i32> {
        let mut vec = Vec::with_capacity(self.dim.0);
        for row in self.data.iter() {
            vec.push(row[idx]);
        }
        vec
    }
}
