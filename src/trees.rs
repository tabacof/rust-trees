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
        train: &Dataset,
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
                    rng = StdRng::seed_from_u64(seed + i as u64);
                } else {
                    rng = StdRng::from_entropy();
                }

                let bootstrap = train.bootstrap(&mut rng);
                TreeNode::_train(
                    &bootstrap,
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
        train: &Dataset,
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
                    rng = StdRng::seed_from_u64(seed + i as u64);
                } else {
                    rng = StdRng::from_entropy();
                }

                let bootstrap = train.bootstrap(&mut rng);
                TreeNode::_train(
                    &bootstrap,
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
            let new_tree = NewTree::from_old_tree(&root, x.feature_names.clone());
            predictions.push(new_tree.predict(x));
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
        train: &Dataset,
        max_depth: i32,
        min_samples_leaf: Option<i32>,
        random_state: Option<u64>,
    ) -> DecisionTree {
        let mut rng = utils::get_rng(random_state);
        let params = TrainOptions {
            max_depth,
            min_samples_leaf: min_samples_leaf
                .unwrap_or(TrainOptions::default_options().min_samples_leaf),
            n_estimators: None,
        };
        DecisionTree {
            root: TreeNode::_train(
                &train,
                0,
                params,
                mean_squared_error_split_feature,
                &mut rng,
            ),
            params,
        }
    }

    #[staticmethod]
    pub fn train_clf(
        train: &Dataset,
        max_depth: i32,
        min_samples_leaf: Option<i32>,
        random_state: Option<u64>,
    ) -> DecisionTree {
        let mut rng = utils::get_rng(random_state);
        let params = TrainOptions {
            max_depth,
            min_samples_leaf: min_samples_leaf
                .unwrap_or(TrainOptions::default_options().min_samples_leaf),
            n_estimators: None,
        };
        DecisionTree {
            root: TreeNode::_train(&train, 0, params, gini_coefficient_split_feature, &mut rng),
            params,
        }
    }

    pub fn predict(&self, test: &Dataset) -> Vec<f32> {
        let index_tree = NewTree::from_old_tree(&self.root, test.feature_names.clone());
        index_tree.predict(test)
    }
}

impl TreeNode {
    fn new_leaf(prediction: f32, n_samples: usize) -> TreeNode {
        TreeNode {
            prediction,
            samples: n_samples,
            split: None,
            feature_name: None,
            left: None,
            right: None,
        }
    }

    fn new_from_split(
        left: TreeNode,
        right: TreeNode,
        split: SplitResult,
        feature_name: &str,
    ) -> TreeNode {
        TreeNode {
            prediction: split.prediction,
            samples: left.samples + right.samples,
            split: Some(split.split),
            feature_name: Some(feature_name.to_string()),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    fn _train(
        train: &Dataset,
        depth: i32,
        train_options: TrainOptions,
        split_feature: SplitFunction,
        rng: &mut StdRng,
    ) -> TreeNode {
        if should_stop(train_options, depth, train) {
            return TreeNode::new_leaf(utils::float_avg(&train.target_vector), train.n_samples());
        }

        let mut best_feature = SplitResult::new_max_loss();
        let mut feature_indexes = (0..train.feature_names.len()).collect::<Vec<usize>>();
        feature_indexes.shuffle(rng);

        for i in feature_indexes {
            if train.feature_uniform[i] {
                continue;
            }

            let split = split_feature(
                i,
                &train.feature_names[i],
                train_options.min_samples_leaf,
                &train.feature_matrix[i],
                &train.target_vector,
            );

            if split.loss < best_feature.loss {
                best_feature = split;
            }
        }

        let (left_ds, right_ds) = split_dataset(&best_feature, train);

        let left_child = TreeNode::_train(&left_ds, depth + 1, train_options, split_feature, rng);
        let right_child = TreeNode::_train(&right_ds, depth + 1, train_options, split_feature, rng);

        let name = &train.feature_names[best_feature.col_index];
        TreeNode::new_from_split(left_child, right_child, best_feature, name)
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

    fn print(&self, depth: usize) {
        match &self.feature_name {
            None => {
                println!(
                    "{:indent$}|-> Leaf: pred: {}, N: {}",
                    "",
                    self.prediction,
                    self.samples,
                    indent = depth * 4
                )
            }
            Some(f) => {
                println!(
                    "{:indent$}-> Branch: feat: {}, th: {}, N: {}, pred: {}",
                    "",
                    f,
                    self.split.unwrap(),
                    self.samples,
                    self.prediction,
                    indent = depth * 4
                );
                self.left.as_ref().unwrap().print(depth + 1);
                self.right.as_ref().unwrap().print(depth + 1);
            }
        }
    }
}

fn split_dataset(split: &SplitResult, dataset: &Dataset) -> (Dataset, Dataset) {
    let mut left_dataset = dataset.clone_without_data();
    let mut right_dataset = dataset.clone_without_data();

    let best_feature_sorter =
        permutation::sort_by(&dataset.feature_matrix[split.col_index], |a, b| {
            a.partial_cmp(b).unwrap_or(Equal)
        });

    for i in 0..dataset.feature_names.len() {
        let sorted_feature = best_feature_sorter.apply_slice(&dataset.feature_matrix[i]);

        let mut first_half = sorted_feature.clone();
        let second_half = first_half.split_off(split.row_index);

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
        &dataset.feature_matrix[split.col_index],
        &dataset.target_vector,
    );

    let mut first_half = sorted_target;
    let second_half = first_half.split_off(split.row_index);

    left_dataset.target_vector = first_half;
    right_dataset.target_vector = second_half;

    (left_dataset, right_dataset)
}

fn should_stop(options: TrainOptions, depth: i32, ds: &Dataset) -> bool {
    let max_depth_reached = depth == options.max_depth;
    let min_samples_reached = options.min_samples_leaf > ds.n_samples() as i32 / 2;
    let uniform_features = ds.feature_uniform.iter().all(|&x| x);
    let one_sample = ds.n_samples() == 1;

    max_depth_reached || min_samples_reached || uniform_features || one_sample
}

// -------------------------------------
// New tree test

type NodeId = usize;
type FeatureIndex = usize;

struct NewTree {
    root: NodeId,
    nodes: Vec<NewNode>,
    feature_names: Vec<String>,
}

enum NewNode {
    Leaf(Leaf),
    Branch(Branch),
}

struct Leaf {
    prediction: f32,
    samples: usize,
}

struct Branch {
    feature: FeatureIndex,
    threshold: f32,
    left: NodeId,
    right: NodeId,
    samples: usize,
    prediction: f32,
}

impl Leaf {
    fn new(prediction: f32, samples: usize) -> Self {
        Leaf {
            prediction,
            samples,
        }
    }
}

impl NewTree {
    fn new(feature_names: Vec<String>) -> Self {
        NewTree {
            root: 0,
            nodes: Vec::new(),
            feature_names,
        }
    }

    fn from_old_tree(root: &TreeNode, feature_names: Vec<String>) -> Self {
        let mut tree = NewTree::new(feature_names);
        tree.root = tree.new_node_from_old(root);
        tree
    }

    fn new_node_from_old(&mut self, old: &TreeNode) -> NodeId {
        let node = match old {
            TreeNode {
                split: None,
                prediction,
                samples,
                ..
            } => NewNode::Leaf(Leaf::new(*prediction, *samples)),
            TreeNode {
                split: Some(threshold),
                prediction,
                samples,
                feature_name: Some(feature_name),
                left: Some(left),
                right: Some(right),
            } => {
                let feature = match self.feature_names.iter().position(|x| x == feature_name) {
                    Some(i) => i,
                    None => {
                        self.feature_names.push(feature_name.clone());
                        self.feature_names.len() - 1
                    }
                };
                let left = self.new_node_from_old(&*left);
                let right = self.new_node_from_old(&*right);
                NewNode::Branch(Branch {
                    feature,
                    threshold: *threshold,
                    left,
                    right,
                    samples: *samples,
                    prediction: *prediction,
                })
            }
            _ => panic!("Invalid Node, either leaf or branch with children expected"),
        };

        self.add_node(node)
    }

    fn print(&self) {
        self.print_node(self.root, 0);
    }

    fn print_node(&self, node: NodeId, depth: usize) {
        match &self.nodes[node] {
            NewNode::Leaf(l) => {
                println!(
                    "{:indent$}|-> Leaf: pred: {}, N: {}",
                    "",
                    l.prediction,
                    l.samples,
                    indent = depth * 4
                );
            }
            NewNode::Branch(b) => {
                println!(
                    "{:indent$}-> Branch: feat: {}, th: {}, N: {}, pred: {}",
                    "",
                    self.feature_names[b.feature],
                    b.threshold,
                    b.samples,
                    b.prediction,
                    indent = depth * 4
                );
                self.print_node(b.left, depth + 1);
                self.print_node(b.right, depth + 1);
            }
        }
    }

    fn add_root(&mut self, node: NewNode) {
        self.nodes.push(node);
        self.root = self.nodes.len() - 1;
    }

    fn add_node(&mut self, node: NewNode) -> NodeId {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn predict(&self, test: &Dataset) -> Vec<f32> {
        let mut predictions = Vec::with_capacity(test.n_samples());
        let mut nodes: Vec<NodeId> = Vec::new();
        for i in 0..test.n_samples() {
            nodes.push(self.root);
            while let Some(node) = nodes.pop() {
                match &self.nodes[node] {
                    NewNode::Leaf(l) => {
                        predictions.push(l.prediction);
                    }
                    NewNode::Branch(b) => {
                        if test.feature_matrix[b.feature][i] < b.threshold {
                            nodes.push(b.left);
                        } else {
                            nodes.push(b.right);
                        }
                    }
                }
            }
            nodes.clear();
        }
        predictions
    }
}

// -------------------------------------

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_predict() {
        let dataset = Dataset::read_csv("datasets/toy_test.csv", ";");

        let root = TreeNode {
            split: Some(2.),
            prediction: 0.5,
            samples: 2,
            feature_name: Some("feature_a".to_string()),
            left: Some(Box::new(TreeNode::new_leaf(1., 1))),
            right: Some(Box::new(TreeNode::new_leaf(0., 1))),
        };

        let dt = DecisionTree {
            root,
            params: TrainOptions {
                max_depth: 1,
                min_samples_leaf: 1,
                n_estimators: None,
            },
        };

        let expected = Dataset::read_csv("datasets/toy_test_predict.csv", ";");
        let pred = dt.predict(&dataset);
        assert_eq!(expected.target_vector, pred);
    }

    #[test]
    fn test_new_predict() {
        let dataset = Dataset::read_csv("datasets/toy_test.csv", ";");

        let root = TreeNode {
            split: Some(2.),
            prediction: 0.5,
            samples: 2,
            feature_name: Some("feature_a".to_string()),
            left: Some(Box::new(TreeNode::new_leaf(1., 1))),
            right: Some(Box::new(TreeNode::new_leaf(0., 1))),
        };

        let dt = DecisionTree {
            root,
            params: TrainOptions {
                max_depth: 1,
                min_samples_leaf: 1,
                n_estimators: None,
            },
        };

        let expected = Dataset::read_csv("datasets/toy_test_predict.csv", ";");
        let pred = dt.predict(&dataset);
        assert_eq!(expected.target_vector, pred);

        let new_tree = NewTree::from_old_tree(&dt.root, dataset.feature_names.clone());
        let new_predictions = new_tree.predict(&dataset);
        assert_eq!(pred, new_predictions);
    }

    #[test]
    fn print_trees() {
        let dataset = Dataset::read_csv("datasets/titanic_train.csv", ",");
        let dt = DecisionTree::train_reg(&dataset, 2, None, None);
        println!("Old Tree");
        dt.root.print(0);

        let new_tree = NewTree::from_old_tree(&dt.root, dataset.feature_names.clone());
        println!("\nNew Tree");
        new_tree.print();
    }
}
