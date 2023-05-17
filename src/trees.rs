use crate::dataset::Dataset;
use crate::split_criteria::gini_coefficient_split_feature;
use crate::split_criteria::mean_squared_error_split_feature;
use crate::split_criteria::SplitFunction;
use crate::split_criteria::SplitResult;
use crate::utils;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;

use pyo3::prelude::*;

#[derive(Debug)]
pub struct TreeNode {
    split: Option<f32>,
    prediction: f32,
    samples: usize,
    feature_name: Option<String>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

#[pyclass]
pub struct DecisionTree {
    root: TreeNode,
    params: TrainOptions,
}

#[pyclass]
pub struct RandomForest {
    roots: Vec<TreeNode>,
    params: TrainOptions,
}

#[derive(Clone, Copy)]
pub struct TrainOptions {
    min_samples_leaf: i32,
    max_depth: i32,
    n_estimators: Option<i32>,
}

impl TrainOptions {
    fn default_options() -> TrainOptions {
        TrainOptions {
            max_depth: 10,
            min_samples_leaf: 1,
            n_estimators: None,
        }
    }
}

#[pymethods]
impl RandomForest {
    #[staticmethod]
    pub fn train_reg(
        train: Dataset,
        n_estimators: i32,
        max_depth: Option<i32>,
        min_samples_leaf: Option<i32>,
        random_state: Option<u64>,
    ) -> RandomForest {
        let params = TrainOptions {
            max_depth: max_depth.unwrap_or(TrainOptions::default_options().max_depth),
            min_samples_leaf: min_samples_leaf
                .unwrap_or(TrainOptions::default_options().min_samples_leaf),
            n_estimators: Some(n_estimators),
        };

        let roots: Vec<TreeNode> = (0..n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng;
                if let Some(seed) = random_state {
                    rng = StdRng::seed_from_u64(seed+i as u64);
                } else {
                    rng = StdRng::from_entropy();
                }
                
                let bootstrap = train.bootstrap(&mut rng);
                TreeNode::_train(
                    bootstrap,
                    0,
                    params,
                    mean_squared_error_split_feature,
                    &mut rng,
                )
            })
            .collect();

        RandomForest { roots, params }
    }

    #[staticmethod]
    pub fn train_clf(
        train: Dataset,
        n_estimators: i32,
        max_depth: Option<i32>,
        min_samples_leaf: Option<i32>,
        random_state: Option<u64>,
    ) -> RandomForest {
        let params = TrainOptions {
            max_depth: max_depth.unwrap_or(TrainOptions::default_options().max_depth),
            min_samples_leaf: min_samples_leaf
                .unwrap_or(TrainOptions::default_options().min_samples_leaf),
            n_estimators: Some(n_estimators),
        };

        let roots: Vec<TreeNode> = (0..n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng;
                if let Some(seed) = random_state {
                    rng = StdRng::seed_from_u64(seed+i as u64);
                } else {
                    rng = StdRng::from_entropy();
                }

                let bootstrap = train.bootstrap(&mut rng);
                TreeNode::_train(
                    bootstrap,
                    0,
                    params,
                    mean_squared_error_split_feature,
                    &mut rng,
                )
            })
            .collect();

        RandomForest { roots, params }
    }

    pub fn predict(&self, x: &Dataset) -> Vec<f32> {
        let mut predictions = Vec::new();
        for root in &self.roots {
            predictions.push(root.predict(x));
        }

        let mut final_predictions = vec![0.0; x.n_samples()];

        for i in 0..x.n_samples() {
            let mut sum = 0.0;
            for j in 0..predictions.len() {
                sum += predictions[j][i];
            }
            final_predictions[i] = sum / predictions.len() as f32;
        }
        final_predictions
    }
}

#[pymethods]
impl DecisionTree {
    #[staticmethod]
    pub fn train_reg(
        train: Dataset,
        max_depth: i32,
        min_samples_leaf: Option<i32>,
        random_state: Option<u64>,
    ) -> DecisionTree {
        let mut rng;
        if let Some(seed) = random_state {
            rng = StdRng::seed_from_u64(seed);
        } else {
            rng = StdRng::from_entropy();
        }
        let params = TrainOptions {
            max_depth,
            min_samples_leaf: min_samples_leaf
                .unwrap_or(TrainOptions::default_options().min_samples_leaf),
            n_estimators: None,
        };
        DecisionTree {
            root: TreeNode::_train(train, 0, params, mean_squared_error_split_feature, &mut rng),
            params,
        }
    }

    #[staticmethod]
    pub fn train_clf(
        train: Dataset,
        max_depth: i32,
        min_samples_leaf: Option<i32>,
        random_state: Option<u64>,
    ) -> DecisionTree {
        let mut rng;
        if let Some(seed) = random_state {
            rng = StdRng::seed_from_u64(seed);
        } else {
            rng = StdRng::from_entropy();
        }
        let params = TrainOptions {
            max_depth,
            min_samples_leaf: min_samples_leaf
                .unwrap_or(TrainOptions::default_options().min_samples_leaf),
            n_estimators: None,
        };
        DecisionTree {
            root: TreeNode::_train(train, 0, params, gini_coefficient_split_feature, &mut rng),
            params,
        }
    }

    pub fn predict(&self, test: &Dataset) -> Vec<f32> {
        self.root.predict(test)
    }
}

impl TreeNode {
    pub fn predict(&self, test: &Dataset) -> Vec<f32> {
        let mut res = vec![];
        for i in 0..test.n_samples() {
            let mut feature_vector = HashMap::new();
            for (j, feature) in test.feature_names.iter().enumerate() {
                feature_vector.insert(feature, test.feature_matrix[j][i]);
            }
            res.push(self.predict_row(&feature_vector));
        }
        res
    }

    fn _train(
        train: Dataset,
        curr_depth: i32,
        train_options: TrainOptions,
        split_feature: SplitFunction,
        rng: &mut StdRng,
    ) -> TreeNode {
        if (curr_depth == train_options.max_depth) | (train.n_samples() == 1)
            || (train_options.min_samples_leaf > train.n_samples() as i32 / 2)
            || (train.feature_uniform.iter().all(|&x| x))
        {
            return TreeNode {
                split: None,
                prediction: utils::float_avg(&train.target_vector),
                samples: train.n_samples(),
                feature_name: None,
                left: None,
                right: None,
            };
        }

        let features = train
            .feature_matrix
            .iter()
            .enumerate()
            .filter(|(index, _)| !train.feature_uniform[*index])
            .map(|(index, feature_vector)| {
                split_feature(
                    index,
                    &train.feature_names[index],
                    train_options.min_samples_leaf,
                    feature_vector,
                    &train.target_vector,
                )
            });

        let loss = features
            .clone()
            .map(|x| x.loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))
            .unwrap();

        let best_features = features
            .filter(|f| f.loss == loss)
            .collect::<Vec<SplitResult>>();

        let best_feature = best_features.choose(rng).unwrap();

        let mut left_dataset = train.clone_without_data();
        let mut right_dataset = train.clone_without_data();

        let best_feature_sorter =
            permutation::sort_by(&train.feature_matrix[best_feature.col_index], |a, b| {
                a.partial_cmp(b).unwrap_or(Equal)
            });

        for i in 0..train.feature_names.len() {
            let sorted_feature = best_feature_sorter.apply_slice(&train.feature_matrix[i]);

            let mut first_half = sorted_feature.clone();
            let second_half = first_half.split_off(best_feature.row_index);

            left_dataset.feature_matrix.push(first_half);
            let first = left_dataset.feature_matrix[i][0];
            left_dataset.feature_uniform[i] =
                left_dataset.feature_matrix[i].iter().all(|&x| x == first);
            right_dataset.feature_matrix.push(second_half);
            let first = right_dataset.feature_matrix[i][0];
            right_dataset.feature_uniform[i] =
                right_dataset.feature_matrix[i].iter().all(|&x| x == first);
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
            samples: train.n_samples(),
            feature_name: Some(train.feature_names[best_feature.col_index].clone()),
            left: Some(Box::new(TreeNode::_train(
                left_dataset,
                curr_depth + 1,
                train_options,
                split_feature,
                rng,
            ))),
            right: Some(Box::new(TreeNode::_train(
                right_dataset,
                curr_depth + 1,
                train_options,
                split_feature,
                rng,
            ))),
        }
    }

    pub fn predict_row(&self, row: &HashMap<&String, f32>) -> f32 {
        if let Some(feature) = &self.feature_name {
            if *row.get(&feature).unwrap() >= self.split.unwrap() {
                self.right
                    .as_ref()
                    .expect("Right node expected")
                    .predict_row(row)
            } else {
                self.left
                    .as_ref()
                    .expect("Left node expected")
                    .predict_row(row)
            }
        } else {
            self.prediction
        }
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
            left: Some(Box::new(TreeNode {
                split: None,
                prediction: 1.,
                samples: 1,
                feature_name: None,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                split: None,
                prediction: 0.,
                samples: 1,
                feature_name: None,
                left: None,
                right: None,
            })),
        };

        let dt = DecisionTree {
            root: root,
            params: TrainOptions {
                max_depth: 1,
                min_samples_leaf: 1,
                n_estimators: None,
            },
        };

        let expected = Dataset::read_csv("datasets/toy_test_predict.csv", ";");
        dt.predict(&mut dataset);
        assert_eq!(expected, dataset);
    }
}
