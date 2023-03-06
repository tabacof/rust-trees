use crate::dataset::Dataset;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::{cell::RefCell, rc::Rc};

#[derive(Debug)]
pub struct TreeNode {
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

    fn split_feature(col_index: usize, feature: &[f32], target: &[f32]) -> SplitResult {
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
                let left_square_sum = square[i - 1];
                let right_square_sum = square[square.len() - 1] - square[i - 1];

                let left_avg = sum[i - 1] / (i as f32);
                let right_avg = (sum[sum.len() - 1] - sum[i - 1]) / (sum.len() - i) as f32;

                let right_mse = right_square_sum - (sum.len() - i) as f32 * right_avg * right_avg;
                let left_mse = left_square_sum - i as f32 * left_avg * left_avg;
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

#[cfg(test)]
mod test {
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
}
