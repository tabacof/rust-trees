use crate::dataset::Dataset;
use crate::split_criteria::gini_coefficient_split_feature;
use crate::split_criteria::mean_squared_error_split_feature;
use crate::split_criteria::SplitFunction;
use crate::split_criteria::SplitResult;
use crate::utils;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
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
        rng: &mut StdRng,
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

        let features = train
            .feature_matrix
            .iter()
            .enumerate()
            .map(|(index, feature_vector)| {
                split_feature(
                    index,
                    &train.feature_names[index],
                    feature_vector,
                    &train.target_vector,
                )
            });

        let loss = features
            .clone()
            .map(|x| x.loss)
            .min_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal))
            .unwrap();

        let best_features = features
            .filter(|f| f.loss == loss)
            .collect::<Vec<SplitResult>>();

        let best_feature = best_features.choose(rng).unwrap();

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
                rng,
            )))),
            right: Some(Rc::new(RefCell::new(TreeNode::_train(
                right_dataset,
                curr_depth + 1,
                max_depth,
                split_feature,
                rng,
            )))),
        }
    }

    pub fn train_reg(train: Dataset, curr_depth: i32, max_depth: i32) -> TreeNode {
        let mut rng = StdRng::seed_from_u64(42);
        Self::_train(
            train,
            curr_depth,
            max_depth,
            mean_squared_error_split_feature,
            &mut rng,
        )
    }

    pub fn train_clf(train: Dataset, curr_depth: i32, max_depth: i32) -> TreeNode {
        let mut rng = StdRng::seed_from_u64(42);
        Self::_train(
            train,
            curr_depth,
            max_depth,
            gini_coefficient_split_feature,
            &mut rng,
        )
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

    #[test]
    fn test_predict() {
        let mut dataset = Dataset::read_csv("datasets/toy_test.csv", ";");
        let root = TreeNode {
            split: Some(2.),
            prediction: 0.5,
            samples: 2,
            feature_name: Some("feature_a".to_string()),
            left: Some(Rc::new(RefCell::new(TreeNode {
                split: None,
                prediction: 0.,
                samples: 1,
                feature_name: None,
                left: None,
                right: None,
            }))),
            right: Some(Rc::new(RefCell::new(TreeNode {
                split: None,
                prediction: 1.,
                samples: 1,
                feature_name: None,
                left: None,
                right: None,
            }))),
        };

        let expected = Dataset::read_csv("datasets/toy_test_predict.csv", ";");
        root.predict(&mut dataset);
        assert_eq!(expected, dataset);
    }
}
