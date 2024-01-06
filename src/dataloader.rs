use csv;

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

    pub fn new_csv(filename: &str, label_index: usize, batch_size: usize, shuffle: bool) -> Self {
        let mut rdr = csv::Reader::from_path(filename).unwrap();
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for result in rdr.records() {
            let record = result.unwrap();
            let mut row = Vec::new();
            for i in 0..record.len() {
                if i == label_index {
                    continue;
                } else {
                    row.push(record[i].parse::<f64>().unwrap());
                }
            }
            data.push(row);
            let mut label = Vec::new();
            label.push(record[label_index].parse::<f64>().unwrap());
            labels.push(label);
        }
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

    pub fn labels_to_categorical(&mut self, num_classes: usize) {
        let mut new_labels = Vec::new();
        for label in self.labels.iter() {
            let mut new_label = vec![0.0; num_classes];
            new_label[label[0] as usize] = 1.0;
            new_labels.push(new_label);
        }
        self.labels = new_labels;
    }

    pub fn normalize_data(&mut self) {
        let mut max = 0.0;
        for row in self.data.iter() {
            for val in row.iter() {
                if *val > max {
                    max = *val;
                }
            }
        }
        for row in self.data.iter_mut() {
            for val in row.iter_mut() {
                *val = *val / max;
            }
        }
    }
}
