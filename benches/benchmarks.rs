use criterion::{criterion_group, criterion_main, Criterion};

use rustrees::r2;
use rustrees::Dataset;
use rustrees::DecisionTree;

fn decision_tree_housing(train: &Dataset, test: &Dataset) {
    let dt = DecisionTree::train_reg(train, 5, Some(1), Some(42));
    if train.n_samples() <= 1 {
        let pred = dt.predict(&test);
        println!("R2: {}", r2(&test.target_vector, &pred));
    }
}

fn predict_decision_tree_housing(dt: &DecisionTree, test: &Dataset) {
    let pred = dt.predict(&test);
    if pred.len() <= 1 {
        println!("R2: {}", r2(&test.target_vector, &pred));
    }
}

fn read_train_test_dataset(name: &str) -> (Dataset, Dataset) {
    let train = "datasets/".to_string() + name + "_train.csv";
    let train = Dataset::read_csv(&train, ",");

    let test = "datasets/".to_string() + name + "_test.csv";
    let test = Dataset::read_csv(&test, ",");

    (train, test)
}
fn criterion_benchmark(c: &mut Criterion) {
    let (train, test) = read_train_test_dataset("housing");
    println!("train: {}", train.n_samples());
    println!("test: {}", test.n_samples());

    // benchmark training
    c.bench_function("train_decision_tree_housing", |b| {
        b.iter(|| decision_tree_housing(&train, &test))
    });

    // benchmark prediction
    let dt = DecisionTree::train_reg(&train, 5, Some(1), Some(42));
    c.bench_function("predict_decision_tree_housing", |b| {
        b.iter(|| predict_decision_tree_housing(&dt, &test))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
