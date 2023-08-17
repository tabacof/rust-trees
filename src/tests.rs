#[cfg(test)]
mod tests {
    use crate::{trees::RandomForest, *};

    fn assert_greater_than(a: f32, b: f32) {
        if a <= b {
            panic!("{} is not greater than {}", a, b);
        }
    }

    #[test]
    fn test_integration() {
        let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
        let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
        let dt = DecisionTree::train_reg(&train, 5, Some(1), Some(42));
        let mut pred = test.clone();
        dt.predict(&mut pred);
        assert_eq!(r2(&test.target_vector, &pred.target_vector) > 0.28, true);
    }

    #[test]
    fn decision_tree_titanic() {
        let (train, test) = read_train_test_dataset("titanic");
        let dt = DecisionTree::train_clf(&train, 5, Some(1), Some(43));
        let pred = dt.predict(&test);
        println!("Accuracy: {}", accuracy(&test.target_vector, &pred));
        assert_greater_than(accuracy(&test.target_vector, &pred), 0.237);
    }

    #[test]
    fn decision_tree_breast_cancer() {
        let (train, test) = read_train_test_dataset("breast_cancer");
        let dt = DecisionTree::train_clf(&train, 5, Some(1), Some(42));
        let pred = dt.predict(&test);
        println!("Accuracy: {}", accuracy(&test.target_vector, &pred));
        assert_greater_than(accuracy(&test.target_vector, &pred), 0.83);
    }

    #[test]
    fn decision_tree_housing() {
        let (train, test) = read_train_test_dataset("housing");
        let dt = DecisionTree::train_reg(&train, 5, Some(1), Some(42));
        let pred = dt.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));
        assert_greater_than(r2(&test.target_vector, &pred), 0.59);
    }

    #[test]
    fn decision_tree_diabeties() {
        let (train, test) = read_train_test_dataset("diabetes");
        let dt = DecisionTree::train_reg(&train, 5, Some(1), Some(42));
        let pred = dt.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));
        assert_greater_than(r2(&test.target_vector, &pred), 0.30);
    }

    fn read_train_test_dataset(name: &str) -> (Dataset, Dataset) {
        let train = "datasets/".to_string() + name + "_train.csv";
        let train = Dataset::read_csv(&train, ",");

        let test = "datasets/".to_string() + name + "_test.csv";
        let test = Dataset::read_csv(&test, ",");

        (train, test)
    }


    #[test]
    fn random_forest_diabetes() {
        let (train, test) = read_train_test_dataset("diabetes");
        let rf = RandomForest::train_reg(&train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));
        assert_greater_than(r2(&test.target_vector, &pred), 0.38);
    }

    #[test]
    fn random_forest_housing() {
        let (train, test) = read_train_test_dataset("housing");
        let rf = RandomForest::train_reg(&train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));
        assert_greater_than(r2(&test.target_vector, &pred), 0.641);
    }

    #[test]
    fn random_forest_breast_cancer() {
        let (train, test) = read_train_test_dataset("breast_cancer");
        let rf = RandomForest::train_clf(&train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        let pred = classification_threshold(&pred, 0.5);

        println!("Accuracy: {}", accuracy(&test.target_vector, &pred),);
        assert_greater_than(accuracy(&test.target_vector, &pred), 0.96);

    }

    #[test]
    fn random_forest_breast_titanic() {
        let (train, test) = read_train_test_dataset("titanic");
        let rf = RandomForest::train_clf(&train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        let pred = classification_threshold(&pred, 0.5);

        println!("Accuracy: {}", accuracy(&test.target_vector, &pred),);
        assert_greater_than(accuracy(&test.target_vector, &pred), 0.789);
    }
}
