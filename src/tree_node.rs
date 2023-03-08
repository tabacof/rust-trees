use crate::dataset::Dataset;
use crate::split_criteria::gini_coefficient_split_feature;
use crate::split_criteria::mean_squared_error_split_feature;
use crate::split_criteria::SplitFunction;
use crate::utils;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::{cell::RefCell, rc::Rc};

#[derive(Debug)]
pub struct TreeNode {
    split: Option<f32>,
    prediction: f32,
    samples: usize,
    feature_name: Option<String>,
    left: Option<Rc<RefCell<TreeNode>>>,
    right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    fn _train(
        train: Dataset,
        curr_depth: i32,
        max_depth: i32,
        split_feature: SplitFunction,
    ) -> TreeNode {
        if (curr_depth == max_depth) | (train.target_vector.len() == 1) {
            return TreeNode {
                split: None,
                prediction: utils::float_avg(&train.target_vector),
                samples: train.target_vector.len(),
                feature_name: None,
                left: None,
                right: None,
            };
        }

        let best_feature = train
            .feature_matrix
            .iter()
            .enumerate()
            .map(|(index, feature_vector)| {
                split_feature(index, feature_vector, &train.target_vector)
            })
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(Equal))
            .unwrap();

        let mut left_dataset = train.clone_without_data();
        let mut right_dataset = train.clone_without_data();

        for i in 0..train.feature_names.len() {
            let (_, sorted_feature) = utils::sort_two_vectors(
                &train.feature_matrix[best_feature.col_index],
                &train.feature_matrix[i],
            );

            let mut first_half = sorted_feature.clone();
            let second_half = first_half.split_off(best_feature.row_index);

            left_dataset.feature_matrix.push(first_half);
            right_dataset.feature_matrix.push(second_half);
        }

        let (_, sorted_target) = utils::sort_two_vectors(
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
            samples: train.target_vector.len(),
            feature_name: Some(train.feature_names[best_feature.col_index].clone()),
            left: Some(Rc::new(RefCell::new(TreeNode::_train(
                left_dataset,
                curr_depth + 1,
                max_depth,
                split_feature,
            )))),
            right: Some(Rc::new(RefCell::new(TreeNode::_train(
                right_dataset,
                curr_depth + 1,
                max_depth,
                split_feature,
            )))),
        }
    }

    pub fn train_reg(train: Dataset, curr_depth: i32, max_depth: i32) -> TreeNode {
        Self::_train(
            train,
            curr_depth,
            max_depth,
            mean_squared_error_split_feature,
        )
    }

    pub fn train_clf(train: Dataset, curr_depth: i32, max_depth: i32) -> TreeNode {
        Self::_train(train, curr_depth, max_depth, gini_coefficient_split_feature)
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
}

#[cfg(test)]
mod test {
    use super::*;
}
