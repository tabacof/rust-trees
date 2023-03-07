mod dataset;
mod tree_node;
mod split_criteria;
mod vec_utils;

use crate::dataset::Dataset;
use crate::tree_node::TreeNode;

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

    println!("Test 5:");
    let train = Dataset::read_csv("datasets/breast_cancer_train.csv", ",");
    let test = Dataset::read_csv("datasets/breast_cancer_test.csv", ",");
    let dt = TreeNode::train_clf(train, 0, 2);
    let mut pred = test.clone();
    dt.predict(&mut pred);

    pred.target_vector = pred
        .target_vector
        .iter()
        .map(|&x| if x >= 0.5 { 1.0 } else { 0.0 })
        .collect();

    println!(
        "Accuracy: {}",
        TreeNode::accuracy(&test.target_vector, &pred.target_vector),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
