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
            .map(String::from)
            .collect();

        let mut feature_matrix: Vec<Vec<f32>> = vec![vec![]; feature_names.len() - 1];
        let mut target_vector: Vec<f32> = Vec::new();

        for (line, row) in split_contents.enumerate() {
            if !row.is_empty() {
                let cols: Vec<&str> = row.split(sep).collect();
                if cols.len() != feature_names.len() {
                    panic!("Wrong number of columns at line {}", line);
                }
                for (i, col) in row.split(sep).enumerate() {
                    let col_val = col.parse::<f32>().unwrap_or(f32::NAN);
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
        clone.feature_matrix = vec![vec![]; clone.feature_names.len() - 1];
        clone.target_vector = vec![];
        clone
    }
}

#[derive(Clone, Debug)]
struct TreeNode {
    split: Option<f32>,
    prediction: f32,
    feature_name: Option<String>,
    left: Option<Rc<RefCell<TreeNode>>>,
    right: Option<Rc<RefCell<TreeNode>>>,
}

struct SplitResult {
    feature_name: String,
    index: usize,
    split: f32,
    prediction: f32,
    loss: f32,
}

impl TreeNode {
    // pub fn new(split: f32, prediction: f32) -> TreeNode {
    //     TreeNode {
    //         split,
    //         prediction,
    //         feature_name,
    //         left: None,
    //         right: None,
    //     }
    // }

    // pub fn add_left(&mut self, left_node: Rc<RefCell<TreeNode>>) {
    //     self.left = Some(left_node);
    // }

    // pub fn add_right(&mut self, right_node: Rc<RefCell<TreeNode>>) {
    //     self.right = Some(right_node);
    // }

    // pub fn split_feature(feature: &Vec<f32>, target: &Vec<f32>) -> (f32, f32, f32) {
    //     let avg: f32 = target.into_iter().sum::<f32>() / target.len() as f32;

    //     let pairs: Vec<(&f32, &f32)> = feature.iter().zip(target.iter()).collect();
    //     pairs.sort_by(|&a, &b| a.0.partial_cmp(b.0).unwrap_or(Equal));

    //     let (sorted_feature, sorted_target): (Vec<f32>, Vec<f32>) =
    //         pairs.into_iter().map(|(x, y)| (*x, *y)).unzip();

    //     let ss = sorted_target.into_iter().map(|x| (x - avg).powf(2.0));

    //     (0.0, 0.0, 0.0)
    // }

    pub fn split_feature(feature: &Vec<f32>, target: &Vec<f32>) -> SplitResult {
        SplitResult {
            feature_name: "feature_a".to_string(),
            index: 0,
            split: 2.0,
            prediction: 3.0,
            loss: 5.0,
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

        let best_feature = train
            .feature_matrix
            .iter()
            .map(|feature| TreeNode::split_feature(feature, &train.target_vector))
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(Equal))
            .unwrap();

        let mut left_dataset = train.clone_without_data();
        let mut right_dataset = train.clone_without_data();

        for row in 0..train.target_vector.len() {
            if train.feature_matrix[best_feature.index][row] > best_feature.split {
                for col in 0..train.feature_names.len() - 1 {
                    right_dataset.feature_matrix[col].push(train.feature_matrix[col][row]);
                }
                right_dataset.target_vector.push(train.target_vector[row]);
            } else {
                for col in 0..train.feature_names.len() - 1 {
                    left_dataset.feature_matrix[col].push(train.feature_matrix[col][row]);
                }
                left_dataset.target_vector.push(train.target_vector[row]);
            }
        }

        TreeNode {
            split: Some(best_feature.split),
            prediction: best_feature.prediction,
            feature_name: Some(best_feature.feature_name),
            left: Some(Rc::new(RefCell::new(TreeNode::train(left_dataset)))),
            right: Some(Rc::new(RefCell::new(TreeNode::train(right_dataset)))),
        }
    }
}

fn main() {
    let train = Dataset::read_csv("train.csv", ";");

    println!("{:#?}", train);

    let dt = TreeNode::train(train);

    println!("{:#?}", dt);
}
