use rayon::prelude::*;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
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

    pub fn write_csv(self, path: &str, sep: &str) {
        let mut contents: String = self.feature_names.join(sep) + sep + &self.target_name + "\n";

        for i in 0..self.target_vector.len() {
            for j in 0..self.feature_names.len() {
                contents += &(self.feature_matrix[j][i].to_string() + sep);
            }
            contents += &(self.target_vector[i].to_string() + "\n");
        }

        fs::write(path, contents).expect("Unable to write file");
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

#[derive(Debug, PartialEq)]
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

    pub fn abserror(x: &[f32]) -> f32 {
        let avg = TreeNode::float_avg(x);

        x.iter().map(|x| (x - avg).abs()).sum()
    }

    pub fn sort_two_vectors(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut pairs: Vec<(&f32, &f32)> = a.iter().zip(b).collect();
        pairs.sort_by(|&a, &b| a.0.partial_cmp(b.0).unwrap_or(Equal));

        pairs.into_iter().map(|(x, y)| (*x, *y)).unzip()
    }

    // pub fn split_feature_mse(col_index: usize, feature: &[f32], target: &[f32]) -> SplitResult {
    //     let (sorted_feature, sorted_target) = TreeNode::sort_two_vectors(feature, target);

    //     let mut row_index = 1;
    //     let mut min_mse = f32::MAX;
    //     let mut last = sorted_feature[0];
    //     let mut left_avg = sorted_target[0];
    //     let mut right_avg = TreeNode::float_avg(&sorted_target[1..]);

    //     for i in 1..sorted_feature.len() {
    //         left_avg = (left_avg * i as f32 + sorted_target[i]) / (i as f32 + 1.0);
    //         right_avg = (right_avg * i as f32 - sorted_target[i])
    //             / (sorted_feature.len() as f32 - i as f32);

    //         if sorted_feature[i] > last {
    //             let mse: f32 = (&sorted_target[0..i])
    //                 .iter()
    //                 .map(|&x| (x - left_avg) * (x - left_avg))
    //                 .sum::<f32>()
    //                 + (&sorted_target[i..])
    //                     .iter()
    //                     .map(|&x| (x - right_avg) * (x - right_avg))
    //                     .sum::<f32>();

    //             if mse <= min_mse {
    //                 row_index = i;
    //                 min_mse = mse;
    //             }

    //             last = sorted_feature[i];
    //         }
    //     }

    //     SplitResult {
    //         col_index,
    //         row_index,
    //         split: sorted_feature[row_index],
    //         prediction: TreeNode::float_avg(target),
    //         loss: min_mse,
    //     }
    // }

    pub fn split_feature(col_index: usize, feature: &[f32], target: &[f32]) -> SplitResult {
        let (sorted_feature, sorted_target) = TreeNode::sort_two_vectors(feature, target);

        let mut row_index = 1;
        let mut min_mse = f32::MAX;
        let mut last = sorted_feature[0];

        let square: Vec<f32> = sorted_target
            .iter()
            .map(|x| x * x)
            .scan(0.0, |state, x| {
                *state = *state + x;
                Some(*state)
            })
            .collect();
        let sum: Vec<f32> = sorted_target
            .iter()
            .scan(0.0, |state, x| {
                *state = *state + x;
                Some(*state)
            })
            .collect();
        for i in 1..sorted_feature.len() {
            if sorted_feature[i] > last {
                //    var = \sum_i^n (y_i - y_bar) ** 2
                //           = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
                //
                let left_square_sum = square[i];
                let right_square_sum = square[square.len() - 1] - square[i];

                let left_avg = sum[i] / (i as f32);
                let right_avg = (sum[sum.len() - 1] - sum[i]) / (sum.len() - i) as f32;

                let right_mse = right_square_sum - (sum.len() - i) as f32 * right_avg * right_avg;
                let left_mse = left_square_sum -  i as f32 * left_avg * left_avg;
                let mse = left_mse + right_mse;

                if mse < min_mse {
                    row_index = i;
                    min_mse = mse;
                }

                last = sorted_feature[i];
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

    pub fn train(train: Dataset, curr_depth: i32, max_depth: i32) -> TreeNode {
        if (curr_depth == max_depth) | (train.target_vector.len() == 1) {
            return TreeNode {
                split: None,
                prediction: TreeNode::float_avg(&train.target_vector),
                feature_name: None,
                left: None,
                right: None,
            };
        }

        let best_feature = train
            .feature_matrix
            .iter()
            //.par_iter()
            .enumerate()
            .map(|(index, feature_vector)| {
                TreeNode::split_feature(index, feature_vector, &train.target_vector)
            })
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(Equal))
            .unwrap();

        let mut left_dataset = train.clone_without_data();
        let mut right_dataset = train.clone_without_data();

        for i in 0..train.feature_names.len() {
            let (_, sorted_feature) = TreeNode::sort_two_vectors(
                &train.feature_matrix[best_feature.col_index],
                &train.feature_matrix[i],
            );

            let mut first_half = sorted_feature.clone();
            let second_half = first_half.split_off(best_feature.row_index);

            left_dataset.feature_matrix.push(first_half);
            right_dataset.feature_matrix.push(second_half);
        }

        let (_, sorted_target) = TreeNode::sort_two_vectors(
            &train.feature_matrix[best_feature.col_index],
            &train.target_vector,
        );

        let mut first_half = sorted_target;
        let second_half = first_half.split_off(best_feature.row_index);

        left_dataset.target_vector = first_half;
        right_dataset.target_vector = second_half;

        TreeNode {
            split: Some(best_feature.split),
            prediction: best_feature.prediction,
            feature_name: Some(train.feature_names[best_feature.col_index].clone()),
            left: Some(Rc::new(RefCell::new(TreeNode::train(
                left_dataset,
                curr_depth + 1,
                max_depth,
            )))),
            right: Some(Rc::new(RefCell::new(TreeNode::train(
                right_dataset,
                curr_depth + 1,
                max_depth,
            )))),
        }
    }

    pub fn predict_row(&self, row: &HashMap<&String, f32>) -> f32 {
        if let Some(feature) = &self.feature_name {
            if *row.get(&feature).unwrap() >= self.split.unwrap() {
                self.right
                    .as_ref()
                    .expect("Right node expected")
                    .borrow()
                    .predict_row(row)
            } else {
                self.left
                    .as_ref()
                    .expect("Left node expected")
                    .borrow()
                    .predict_row(row)
            }
        } else {
            self.prediction
        }
    }

    pub fn predict(&self, test: &mut Dataset) {
        let mut res = vec![];
        for i in 0..test.target_vector.len() {
            let mut feature_vector = HashMap::new();
            for (j, feature) in test.feature_names.iter().enumerate() {
                feature_vector.insert(feature, test.feature_matrix[j][i]);
            }
            res.push(self.predict_row(&feature_vector));
        }
        test.target_vector = res;
    }

    pub fn r2(x_true: &[f32], x_pred: &[f32]) -> f32 {
        let mse: f32 = x_true
            .iter()
            .zip(x_pred)
            .map(|(xt, xp)| (xt - xp).powf(2.0))
            .sum();

        let avg = TreeNode::float_avg(x_true);
        let var: f32 = x_true.iter().map(|x| (x - avg).powf(2.0)).sum();

        1.0 - mse / var
    }
}

fn main() {
    println!("Test 1:");
    let train = Dataset::read_csv("datasets/toy_train.csv", ";");
    let mut test = Dataset::read_csv("datasets/toy_test.csv", ";");
    let dt = TreeNode::train(train, 0, 4);
    println!("{:#?}", dt);
    dt.predict(&mut test);
    println!("{:#?}", test);

    println!("Test 2:");
    let train = Dataset::read_csv("datasets/one_feature_train.csv", ",");
    let mut test = Dataset::read_csv("datasets/one_feature_test.csv", ",");
    let dt = TreeNode::train(train, 0, 5);
    dt.predict(&mut test);
    println!("{:#?}", test);

    println!("Test 3:");
    let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
    let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
    let dt = TreeNode::train(train, 0, 5);
    let mut pred = test.clone();
    dt.predict(&mut pred);
    println!(
        "R2: {}",
        TreeNode::r2(&test.target_vector, &pred.target_vector),
    );

    println!("Test 4:");
    let train = Dataset::read_csv("datasets/housing_train.csv", ",");
    let test = Dataset::read_csv("datasets/housing_test.csv", ",");
    let dt = TreeNode::train(train, 0, 10);
    let mut pred = test.clone();
    dt.predict(&mut pred);
    println!(
        "R2: {}",
        TreeNode::r2(&test.target_vector, &pred.target_vector),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_avg() {
        assert_eq!(TreeNode::float_avg(&vec![-1.0, 0.0, 1.0]), 0.0);
        assert_eq!(TreeNode::float_avg(&vec![1.0]), 1.0);
    }

    #[test]
    fn test_mse() {
        assert_eq!(TreeNode::mse(&vec![-1.0, 0.0, 1.0]), 2.0);
        assert_eq!(TreeNode::mse(&vec![1.0]), 0.0);
    }

    #[test]
    fn test_sort_two_vectors() {
        assert_eq!(
            TreeNode::sort_two_vectors(&vec![2.0, 0.0, 1.0], &vec![-1.0, 0.0, 1.0]),
            (vec![0.0, 1.0, 2.0], vec![0.0, 1.0, -1.0])
        );
    }

    #[test]
    fn test_split_feature() {
        assert_eq!(
            TreeNode::split_feature(1, &vec![2.0, 0.0, 1.0], &vec![-1.0, 0.0, 1.0]),
            SplitResult {
                col_index: 1,
                row_index: 2,
                split: 2.0,
                prediction: 0.0,
                loss: 0.5,
            }
        );
    }

    #[test]
    fn test_integration() {
        let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
        let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
        let dt = TreeNode::train(train, 0, 5);
        let mut pred = test.clone();
        dt.predict(&mut pred);
        assert_eq!(
            TreeNode::r2(&test.target_vector, &pred.target_vector) > 0.28,
            true
        );
    }
}
