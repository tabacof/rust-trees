#[cfg(test)]
mod tests {
    use crate::{trees::RandomForest, *};

    #[test]
    fn test_integration() {
        let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
        let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
        let dt = DecisionTree::train_reg(train, 5, Some(1), Some(42));
        let mut pred = test.clone();
        dt.predict(&mut pred);
        assert_eq!(r2(&test.target_vector, &pred.target_vector) > 0.28, true);
    }

    #[test]
    fn test_all_datasets_dt() {
        println!("Test 1: toy regression");
        let train = Dataset::read_csv("datasets/toy_train.csv", ";");
        let test = Dataset::read_csv("datasets/toy_test.csv", ";");
        let dt = DecisionTree::train_reg(train, 2, Some(1), Some(42));
        dt.predict(&test);

        println!("Test 2: one feature regression");
        let train = Dataset::read_csv("datasets/one_feature_train.csv", ",");
        let test = Dataset::read_csv("datasets/one_feature_test.csv", ",");
        let dt = DecisionTree::train_reg(train, 2, Some(1), Some(42));
        dt.predict(&test);

        println!("Test 3: diabetes regression");
        let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
        let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
        let dt = DecisionTree::train_reg(train, 5, Some(1), Some(42));
        let pred = dt.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));

        println!("Test 4: housing regression");
        let train = Dataset::read_csv("datasets/housing_train.csv", ",");
        let test = Dataset::read_csv("datasets/housing_test.csv", ",");
        let dt = DecisionTree::train_reg(train, 5, Some(1), Some(42));
        let pred = dt.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));

        println!("Test 5: breast cancer classification");
        let train = Dataset::read_csv("datasets/breast_cancer_train.csv", ",");
        let test = Dataset::read_csv("datasets/breast_cancer_test.csv", ",");
        let dt = DecisionTree::train_clf(train, 5, Some(1), Some(42));
        let pred = dt.predict(&test);
        let pred = classification_threshold(&pred, 0.5);

        println!("Accuracy: {}", accuracy(&test.target_vector, &pred),);

        println!("Test 6: Titanic classification");
        let train = Dataset::read_csv("datasets/titanic_train.csv", ",");
        let test = Dataset::read_csv("datasets/titanic_test.csv", ",");
        let dt = DecisionTree::train_clf(train, 5, Some(1), Some(42));
        let pred = dt.predict(&test);
        let pred = classification_threshold(&pred, 0.5);

        println!("Accuracy: {}", accuracy(&test.target_vector, &pred),);
    }

    #[test]
    fn test_all_datasets_rf() {
        println!("Test 1: diabetes regression");
        let train = Dataset::read_csv("datasets/diabetes_train.csv", ",");
        let test = Dataset::read_csv("datasets/diabetes_test.csv", ",");
        let rf = RandomForest::train_reg(train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));

        println!("Test 2: housing regression");
        let train = Dataset::read_csv("datasets/housing_train.csv", ",");
        let test = Dataset::read_csv("datasets/housing_test.csv", ",");
        let rf = RandomForest::train_reg(train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));

        println!("Test 3: breast cancer classification");
        let train = Dataset::read_csv("datasets/breast_cancer_train.csv", ",");
        let test = Dataset::read_csv("datasets/breast_cancer_test.csv", ",");
        let rf = RandomForest::train_clf(train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        let pred = classification_threshold(&pred, 0.5);

        println!("Accuracy: {}", accuracy(&test.target_vector, &pred),);

        println!("Test 4: Titanic classification");
        let train = Dataset::read_csv("datasets/titanic_train.csv", ",");
        let test = Dataset::read_csv("datasets/titanic_test.csv", ",");
        let rf = RandomForest::train_clf(train, 10, Some(5), Some(1), Some(42));
        let pred = rf.predict(&test);
        let pred = classification_threshold(&pred, 0.5);

        println!("Accuracy: {}", accuracy(&test.target_vector, &pred),);
    }
}
