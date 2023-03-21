mod dataset;
mod split_criteria;
mod tree_node;
mod utils;

use crate::dataset::Dataset;
use crate::tree_node::TreeNode;

fn main() {
    println!("Test 1: toy regression");
    let train = Dataset::read_csv("datasets/toy_train.csv", ";");
    let mut test = Dataset::read_csv("datasets/toy_test.csv", ";");
    let dt = TreeNode::train_reg(train, 0, 2);
    dt.predict(&mut test);

    println!("Test 2: one feature regression");
    let train = Dataset::read_csv("datasets/one_feature_train.csv", ",");
    let mut test = Dataset::read_csv("datasets/one_feature_test.csv", ",");
    let dt = TreeNode::train_reg(train, 0, 2);
    dt.predict(&mut test);

    println!("Test 3: diabetes regression");
    let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
    let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
    let dt = TreeNode::train_reg(train, 0, 5);
    let mut pred = test.clone();
    dt.predict(&mut pred);
    println!(
        "R2: {}",
        utils::r2(&test.target_vector, &pred.target_vector),
    );

    println!("Test 4: housing regression");
    let train = Dataset::read_csv("datasets/housing_train.csv", ",");
    let test = Dataset::read_csv("datasets/housing_test.csv", ",");
    let dt = TreeNode::train_reg(train, 0, 5);
    let mut pred = test.clone();
    dt.predict(&mut pred);
    println!(
        "R2: {}",
        utils::r2(&test.target_vector, &pred.target_vector),
    );

    println!("Test 5: breast cancer classification");
    let train = Dataset::read_csv("datasets/breast_cancer_train.csv", ",");
    let test = Dataset::read_csv("datasets/breast_cancer_test.csv", ",");
    let dt = TreeNode::train_clf(train, 0, 5);
    let mut pred = test.clone();
    dt.predict(&mut pred);

    pred.target_vector = utils::classification_threshold(&pred.target_vector, 0.5);

    println!(
        "Accuracy: {}",
        utils::accuracy(&test.target_vector, &pred.target_vector),
    );

    println!("Test 6: Titanic classification");
    let train = Dataset::read_csv("datasets/titanic_train.csv", ",");
    let test = Dataset::read_csv("datasets/titanic_test.csv", ",");
    let dt = TreeNode::train_clf(train, 0, 5);
    let mut pred = test.clone();
    dt.predict(&mut pred);

    pred.target_vector = utils::classification_threshold(&pred.target_vector, 0.5);

    println!(
        "Accuracy: {}",
        utils::accuracy(&test.target_vector, &pred.target_vector),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration() {
        let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
        let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
        let dt = TreeNode::train_reg(train, 0, 5);
        let mut pred = test.clone();
        dt.predict(&mut pred);
        assert_eq!(
            utils::r2(&test.target_vector, &pred.target_vector) > 0.28,
            true
        );
    }
}
