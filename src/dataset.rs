use arrow::array::Float32Array;
use arrow::compute::cast;
use arrow::csv;
use arrow::datatypes::DataType;
use arrow::pyarrow::PyArrowConvert;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use rand::{rngs::StdRng, Rng};
use std::fs;
use std::fs::File;

use pyo3::types::PyAny;

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    pub feature_names: Vec<String>,
    pub feature_uniform: Vec<bool>,
    pub feature_matrix: Vec<Vec<f32>>,
    pub target_name: String,
    pub target_vector: Vec<f32>,
}

impl Dataset {
    fn _from_pyarrow(df: &PyAny) -> Dataset {
        let batch = RecordBatch::from_pyarrow(df).unwrap();

        let feature_names = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<String>>();

        let feature_matrix: Vec<Vec<f32>> = Dataset::_read_batch(batch);

        Dataset {
            feature_names: feature_names[0..feature_names.len() - 1].to_vec(),
            feature_uniform: vec![false; feature_names.len() - 1],
            feature_matrix: feature_matrix[0..feature_matrix.len() - 1].to_vec(),
            target_name: feature_names.last().unwrap().to_string(),
            target_vector: feature_matrix.last().unwrap().to_vec(),
        }
    }

    // usefull for tests
    pub(crate) fn from_vecs<S: Into<String>>(
        col_names: Vec<S>,
        row_vecs: Vec<Vec<f32>>,
    ) -> Dataset {
        assert!(row_vecs.len() > 0);
        assert!(col_names.len() > 0);
        assert!(col_names.len() == row_vecs[0].len());

        let feature_names: Vec<String> = col_names[0..col_names.len() - 1]
            .iter()
            .map(|x| x.into().clone())
            .collect();
        let target_name = col_names[col_names.len() - 1].into();
        let row_len = row_vecs[0].len();

        // create feature matrix
        // create target vector
        // create one vector per feature with capacity equal to samples
        // for each row
        //  push value to target vector
        //  for each feature
        //  push value to feature vector
        let mut feature_matrix = Vec::with_capacity(feature_names.len());
        let mut target_vector = Vec::with_capacity(row_vecs.len());
        for _ in 0..feature_names.len() {
            feature_matrix.push(Vec::with_capacity(row_vecs.len()));
        }
        for i in 0..row_vecs.len() {
            target_vector.push(row_vecs[i][row_len - 1]);
            for j in 0..feature_names.len() {
                feature_matrix[j].push(row_vecs[i][j]);
            }
        }

        Dataset {
            feature_names,
            feature_uniform: vec![false; col_names.len() - 1],
            feature_matrix,
            target_name,
            target_vector,
        }
    }

    fn _read_batch(batch: RecordBatch) -> Vec<Vec<f32>> {
        batch
            .columns()
            .iter()
            .map(|c| cast(c, &DataType::Float32).unwrap())
            .map(|c| {
                c.as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>()
    }

    fn _read_csv(path: &str, sep: &str) -> Dataset {
        let file = File::open(path).unwrap();
        let builder = csv::ReaderBuilder::new()
            .with_delimiter(sep.as_bytes()[0])
            .has_header(true)
            .infer_schema(Some(100));
        let mut csv = builder.build(file).unwrap();
        let batch = csv.next().unwrap().unwrap();

        let feature_names = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<String>>();

        let mut feature_matrix: Vec<Vec<f32>> = Dataset::_read_batch(batch);

        for b in csv {
            let batch = b.unwrap();
            let mut batch_matrix = Dataset::_read_batch(batch);
            for i in 0..feature_matrix.len() {
                feature_matrix[i].append(&mut batch_matrix[i]);
            }
        }

        Dataset {
            feature_names: feature_names[0..feature_names.len() - 1].to_vec(),
            feature_uniform: vec![false; feature_names.len() - 1],
            feature_matrix: feature_matrix[0..feature_matrix.len() - 1].to_vec(),
            target_name: feature_names.last().unwrap().to_string(),
            target_vector: feature_matrix.last().unwrap().to_vec(),
        }
    }

    pub fn clone_without_data(&self) -> Dataset {
        Dataset {
            feature_names: self.feature_names.clone(),
            feature_uniform: vec![false; self.feature_names.len()],
            feature_matrix: vec![],
            target_name: self.target_name.clone(),
            target_vector: vec![],
        }
    }

    pub fn n_samples(&self) -> usize {
        self.target_vector.len()
    }

    pub fn bootstrap(&self, rng: &mut StdRng) -> Dataset {
        let mut feature_matrix: Vec<Vec<f32>> = vec![vec![]; self.feature_names.len()];
        let mut target_vector: Vec<f32> = Vec::new();

        for _ in 0..self.target_vector.len() {
            let i = rng.gen_range(0..self.target_vector.len());

            for j in 0..self.feature_names.len() {
                feature_matrix[j].push(self.feature_matrix[j][i]);
            }
            target_vector.push(self.target_vector[i]);
        }

        Dataset {
            feature_names: self.feature_names.clone(),
            feature_uniform: self.feature_uniform.clone(),
            feature_matrix,
            target_name: self.target_name.clone(),
            target_vector,
        }
    }
}

#[pymethods]
impl Dataset {
    #[staticmethod]
    pub fn read_csv(path: &str, sep: &str) -> Dataset {
        println!("Reading CSV file {}", path);
        //let contents = fs::read_to_string(path).expect("Cannot read CSV file");
        Self::_read_csv(path, sep)
    }

    pub fn write_csv(&self, path: &str, sep: &str) {
        let mut contents: String = self.feature_names.join(sep) + sep + &self.target_name + "\n";

        for i in 0..self.target_vector.len() {
            for j in 0..self.feature_names.len() {
                contents += &(self.feature_matrix[j][i].to_string() + sep);
            }
            contents += &(self.target_vector[i].to_string() + "\n");
        }

        fs::write(path, contents).expect("Unable to write file");
    }

    #[staticmethod]
    pub fn from_pyarrow(df: &PyAny) -> Dataset {
        Self::_from_pyarrow(df)
    }

    pub fn add_target(&mut self, target: Vec<f32>) {
        self.target_vector = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_csv() {
        let got = Dataset::read_csv("datasets/toy_test.csv", ";");

        let expected = Dataset {
            feature_names: vec!["feature_a".to_string(), "feature_b".to_string()],
            feature_uniform: vec![false, false],
            feature_matrix: vec![vec![1., 3.], vec![2., 4.]],
            target_name: "target".to_string(),
            target_vector: vec![1.0, 0.0],
        };

        assert_eq!(expected, got);
    }

    #[test]
    fn test_clone_without_data() {
        let dataset = Dataset {
            feature_names: vec!["age".to_string(), "sex".to_string()],
            feature_uniform: vec![false, false],
            feature_matrix: vec![vec![0.1, 1.0], vec![-0.5, 2.0]],
            target_name: "target".to_string(),
            target_vector: vec![5.0, 3.0],
        };

        let expected = Dataset {
            feature_names: vec!["age".to_string(), "sex".to_string()],
            feature_uniform: vec![false, false],
            feature_matrix: vec![],
            target_name: "target".to_string(),
            target_vector: vec![],
        };

        let got = dataset.clone_without_data();

        assert_eq!(expected, got);
    }

    #[test]
    fn test_dataset_print() {
        let ds = Dataset::from_vecs(
            vec!["feature_a", "feature_b", "target"],
            vec![vec![1., 3., 0.], vec![2., 4., 0.]],
        );
    }
}
