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

/// Dataset represents the data used to train the model.
///
/// Data is stored grouped by features (columns) and the target is stored separately.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    /// Keeping the feature_names allows for mapping feature_indexes to feature names
    pub feature_names: Vec<String>,
    /// feature_uniform is a vector of booleans that indicates whether a feature is uniform.
    /// Uniform features are never useful for splitting so we can skip them. Also, once a feature
    /// is uniform it will always be uniform for further splits.
    pub feature_uniform: Vec<bool>,

    /// feature_matrix is a vector of features, where each feature is a vector of values.
    /// The algorithm oftenn iterates over the features, so it is more efficient to keep
    /// them in a vector of features, rather than a vector of rows.
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

    pub(crate) fn clone_without_data(&self) -> Dataset {
        Dataset {
            feature_names: self.feature_names.clone(),
            feature_uniform: vec![false; self.feature_names.len()],
            feature_matrix: vec![],
            target_name: self.target_name.clone(),
            target_vector: vec![],
        }
    }

    /// exposes the size of the dataset
    pub fn n_samples(&self) -> usize {
        self.target_vector.len()
    }

    pub(crate) fn bootstrap(&self, rng: &mut StdRng) -> Dataset {
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

/// Methods exposed to the python binding.
#[pymethods]
impl Dataset {

    /// Reads a CSV file and returns a Dataset.
    #[staticmethod]
    pub fn read_csv(path: &str, sep: &str) -> Dataset {
        println!("Reading CSV file {}", path);
        //let contents = fs::read_to_string(path).expect("Cannot read CSV file");
        Self::_read_csv(path, sep)
    }

    /// Writes dataset to a CSV file.
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

    /// Converts a pyarrow datafram to a Dataset.
    /// This is used to convert a pandas dataframe to a Dataset.
    #[staticmethod]
    pub fn from_pyarrow(df: &PyAny) -> Dataset {
        Self::_from_pyarrow(df)
    }

    /// Adds the target vector to the dataset.
    /// A dataset always has the feature matrix, but will only have the target vector 
    /// when it is for training.
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
}
