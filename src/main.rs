mod dataset;
mod tree_node;

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
