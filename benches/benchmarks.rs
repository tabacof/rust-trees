use std::num::NonZeroUsize;

use criterion::{criterion_group, criterion_main, Criterion};
use criterion::black_box;

use rustrees::r2;
use rustrees::Dataset;
use rustrees::DecisionTree;

use randomforest::criterion::Mse;
use randomforest::RandomForestRegressorOptions;
use randomforest::table::{Table, TableBuilder};

fn decision_tree_housing(train: &Dataset, test: &Dataset) {
    let dt = DecisionTree::train_reg(train, 14, Some(1), Some(42));
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


fn load_randomforest_dataset<'a>(table_builder: &'a mut TableBuilder, train: &'a Dataset) -> Table<'a> {
    let features = &train.feature_matrix;
    let target = &train.target_vector;

    for i in 0..target.len() {
        let mut x = Vec::new();
        for j in 0..features.len() {
            x.push(features[j][i] as f64);
        }
        table_builder.add_row(&x, target[i] as f64).unwrap();
    }
    table_builder.build().unwrap()
}

fn randomforest(table: Table, num_features: usize) {
    let regressor = RandomForestRegressorOptions::new()
        .seed(0)
        .trees(NonZeroUsize::new(1).unwrap())
        .max_features(NonZeroUsize::new(num_features).unwrap())
        .fit(Mse, table);
}

fn criterion_benchmark(c: &mut Criterion) {
    let dataset = "housing";
    let (train, test) = read_train_test_dataset(dataset);
    println!("train: {}", train.n_samples());
    println!("test: {}", test.n_samples());

    // benchmark training
    let train_name = "train_decision_tree_".to_string() + dataset;
    c.bench_function(&train_name, |b| {
        b.iter(|| decision_tree_housing(&train, &test))
    });

    // benchmark prediction
    // let pred_name = "predict_decision_tree_".to_string() + dataset;
    // let dt = DecisionTree::train_reg(&train, 5, Some(1), Some(42));
    // c.bench_function(&pred_name, |b| {
    //     b.iter(|| predict_decision_tree_housing(&dt, &test))
    // });

    // randomforest crate benchmark
    let mut table_builder = TableBuilder::new();
    let table = load_randomforest_dataset(&mut table_builder, &train);
    
    let pred_name: String = "clone_housing".to_string();
    c.bench_function(&pred_name, |b| {
        b.iter(|| {
            let table_arg: Table<'_> = table.clone();
            black_box(table_arg)
        })
    });

    let pred_name = "randomforest_crate".to_string();
    c.bench_function(&pred_name, |b| {
        b.iter(|| {
            let table_arg: Table<'_> = table.clone();
            randomforest(table_arg, train.feature_names.len())
        })
    });


}



criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
