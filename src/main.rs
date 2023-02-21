use std::cmp::Ordering::Equal;
use std::fs;
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug)]
struct Dataset {
    feature_names: Vec<String>,
    feature_matrix: Vec<Vec<f32>>,
    target_name: String,
    target_vector: Vec<f32>,
}

impl Dataset {
    pub fn read_csv(path: &str, sep: &str) -> Dataset {
        let contents = fs::read_to_string(path).expect("Cannot read CSV file");

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

        Dataset {
            feature_names,
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
}

#[derive(Debug)]
struct TreeNode {
    split: Option<f32>,
    prediction: f32,
    feature_name: Option<String>,
    left: Option<Rc<RefCell<TreeNode>>>,
    right: Option<Rc<RefCell<TreeNode>>>,
}

#[derive(Debug)]
struct SplitResult {
    col_index: usize,
    row_index: usize,
    split: f32,
    prediction: f32,
    loss: f32,
}

impl TreeNode {
    pub fn float_avg(x: &[f32]) -> f32 {
        x.iter().sum::<f32>() / x.len() as f32
    }

    pub fn mse(x: &[f32]) -> f32 {
        let avg = TreeNode::float_avg(x);

        x.iter().map(|x| (x - avg).powf(2.0)).sum()
    }

    pub fn split_feature(col_index: usize, feature: &[f32], target: &[f32]) -> SplitResult {
        let mut pairs: Vec<(&f32, &f32)> = feature.iter().zip(target.iter()).collect();
        pairs.sort_by(|&a, &b| a.0.partial_cmp(b.0).unwrap_or(Equal));

        let (sorted_feature, sorted_target): (Vec<f32>, Vec<f32>) =
            pairs.into_iter().map(|(x, y)| (*x, *y)).unzip();

        let mut row_index = 1;
        let mut min_mse = f32::MAX;

        for i in 1..sorted_feature.len() {
            let mut first_half = sorted_target.clone();
            let second_half = first_half.split_off(i);

            let mse = TreeNode::mse(&first_half) + TreeNode::mse(&second_half);
            println!("mse {}", mse);

            if mse <= min_mse {
                row_index = i;
                min_mse = mse;
            }
        }

        SplitResult {
            col_index,
            row_index,
            split: sorted_feature[row_index],
            prediction: TreeNode::float_avg(target),
            loss: min_mse,
        }
    }

    pub fn train(train: Dataset) -> TreeNode {
        if train.target_vector.len() == 1 {
            return TreeNode {
                split: None,
                prediction: train.target_vector[0],
                feature_name: None,
                left: None,
                right: None,
            };
        }

        println!("train={:?}", train);

        let best_feature = train
            .feature_matrix
            .iter()
            .enumerate()
            .map(|(index, feature_vector)| {
                TreeNode::split_feature(index, feature_vector, &train.target_vector)
            })
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(Equal))
            .unwrap();

        println!("best_feature={:?}", best_feature);

        let mut left_dataset = train.clone_without_data();
        let mut right_dataset = train.clone_without_data();

        for i in 0..train.feature_names.len() {
            if i != best_feature.col_index {
                let mut pairs: Vec<(&f32, &f32)> = train.feature_matrix[best_feature.col_index]
                    .iter()
                    .zip(train.feature_matrix[i].iter())
                    .collect();
                pairs.sort_by(|&a, &b| a.0.partial_cmp(b.0).unwrap_or(Equal));

                let (_, sorted_feature): (Vec<f32>, Vec<f32>) =
                    pairs.into_iter().map(|(x, y)| (*x, *y)).unzip();

                let mut first_half = sorted_feature.clone();
                let second_half = first_half.split_off(best_feature.row_index);

                left_dataset.feature_matrix.push(first_half);
                right_dataset.feature_matrix.push(second_half);
            } else {
                let mut first_half = train.feature_matrix[best_feature.col_index].clone();
                let second_half = first_half.split_off(best_feature.row_index);

                left_dataset.feature_matrix.push(first_half);
                right_dataset.feature_matrix.push(second_half);
            }
        }

        let mut pairs: Vec<(&f32, &f32)> = train.feature_matrix[best_feature.col_index]
            .iter()
            .zip(train.target_vector.iter())
            .collect();
        pairs.sort_by(|&a, &b| a.0.partial_cmp(b.0).unwrap_or(Equal));

        let (_, sorted_target): (Vec<f32>, Vec<f32>) =
            pairs.into_iter().map(|(x, y)| (*x, *y)).unzip();

        let mut first_half = sorted_target;
        let second_half = first_half.split_off(best_feature.row_index);

        left_dataset.target_vector = first_half;
        right_dataset.target_vector = second_half;

        println!("left: {:?}", left_dataset);

        println!("right: {:?}", right_dataset);

        TreeNode {
            split: Some(best_feature.split),
            prediction: best_feature.prediction,
            feature_name: Some(train.feature_names[best_feature.col_index].clone()),
            left: Some(Rc::new(RefCell::new(TreeNode::train(left_dataset)))),
            right: Some(Rc::new(RefCell::new(TreeNode::train(right_dataset)))),
        }
    }
}

fn main() {
    // let train = Dataset::read_csv("datasets/toy.csv", ";");

    // println!("{:#?}", train);

    // let dt = TreeNode::train(train);

    // println!("{:#?}", dt);

    let train = Dataset::read_csv("datasets/one_feature.csv", ",");

    println!("{:#?}", train);

    let dt = TreeNode::train(train);

    println!("{:#?}", dt);
}
