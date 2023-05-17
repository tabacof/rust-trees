use pyo3::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs;
use arrow::array::{ArrayRef, Float32Array, StringArray, UInt32Array};
use arrow::compute::cast;
use arrow::csv;
use arrow::error::Result;
use arrow::util::pretty::print_batches;
use std::fs::File;
use std::iter::IntoIterator;
use arrow::datatypes::DataType;



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

        let feature_matrix: Vec<Vec<f32>> = batch
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
            .collect::<Vec<_>>();

        Dataset {
            feature_names: feature_names[0..feature_names.len() - 1].to_vec(),
            feature_uniform: vec![false; feature_names.len() - 1],
            feature_matrix: feature_matrix[0..feature_matrix.len() - 1].to_vec(),
            target_name: feature_names.last().unwrap().to_string(),
            target_vector: feature_matrix.last().unwrap().to_vec(),
        }
    }
    fn _read_csv2(contents: &str, sep: &str) -> Dataset {
        let mut split_contents = contents.split('\n');

        let mut feature_names: Vec<String> = split_contents
            .next()
            .expect("Cannot read columns")
            .split(sep)
            .map(str::trim)
            .map(String::from)
            .collect();

        let mut feature_matrix: Vec<Vec<f32>> = vec![vec![]; feature_names.len() - 1];
        let mut target_vector: Vec<f32> = Vec::new();

        for (line, row) in split_contents.enumerate() {
            if !row.is_empty() {
                let cols = row.split(sep);
                if cols.count() != feature_names.len() {
                    panic!("Wrong number of columns at line {}", line);
                }
                for (i, col) in row.split(sep).enumerate() {
                    let col_val = col.trim().parse::<f32>().unwrap_or(f32::NAN);
                    if i == feature_names.len() - 1 {
                        target_vector.push(col_val);
                    } else {
                        feature_matrix[i].push(col_val);
                    }
                }
            }
        }

        let target_name = feature_names.pop().expect("We need at least one column");
        let feature_uniform = vec![false; feature_names.len()];

        Dataset {
            feature_names,
            feature_uniform,
            feature_matrix,
            target_name,
            target_vector,
        }
    }

    pub fn clone_without_data(&self) -> Dataset {
        let mut clone = self.clone();
        clone.feature_matrix = vec![];
        clone.target_vector = vec![];
        clone
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
