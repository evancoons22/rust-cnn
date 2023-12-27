pub struct DataLoader {
    pub data: Vec<Vec<f64>>,
    pub labels: Vec<Vec<f64>>,
    pub batch_size: usize,
    pub shuffle: bool,
    pub batch_index: usize,
    pub num_batches: usize,
}

impl DataLoader {
    pub fn new(data: Vec<Vec<f64>>, labels: Vec<Vec<f64>>, batch_size: usize, shuffle: bool) -> Self {
        let num_batches = data.len() / batch_size;
        DataLoader {
            data,
            labels,
            batch_size,
            shuffle,
            batch_index: 0,
            num_batches,
        }
    }

    pub fn next_batch(&mut self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut batch_data = Vec::new();
        let mut batch_labels = Vec::new();
        for i in 0..self.batch_size {
            batch_data.push(self.data[self.batch_index * self.batch_size + i].clone());
            batch_labels.push(self.labels[self.batch_index * self.batch_size + i].clone());
        }
        self.batch_index += 1;
        if self.batch_index == self.num_batches {
            self.batch_index = 0;
        }
        (batch_data, batch_labels)
    }
}
