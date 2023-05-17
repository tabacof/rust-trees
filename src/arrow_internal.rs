use crate::dataset::Dataset;
use arrow::array::{ArrayRef, Float32Array, StringArray, UInt32Array};
use arrow::compute::cast;
use arrow::csv;
use arrow::datatypes::DataType;
use arrow::error::Result;
use arrow::util::pretty::print_batches;
use std::fs::File;
use std::iter::IntoIterator;

fn read_from_csv(path: &str, sep: &str) -> Dataset {
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



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_arrow_to_data_set() {
        let dataset = read_from_csv("datasets/toy_test.csv", ";");
        let dataset2 = Dataset::read_csv("datasets/toy_test.csv", ";");

        assert_eq!(dataset, dataset2);
    }
}
