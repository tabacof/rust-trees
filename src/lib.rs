//! Rustrees is a library for building decision trees and random forests.
//!
//! The goal is to provide a fast implementation of decision trees in rust, with a python API.
//!
//! Example usage:
//!
//! ```rust
//! use rustrees::{DecisionTree, Dataset, r2};
//! 
//! let dataset = Dataset::read_csv("datasets/titanic_train.csv", ",");
//! 
//! let dt = DecisionTree::train_reg(
//!    &dataset, 
//!    Some(5),        // max_depth
//!    Some(1),  // min_samples_leaf        
//!    None,     // max_features (None = all features)
//!    Some(42), // random_state
//! );
//! 
//! let pred = dt.predict(&dataset);
//! 
//! println!("r2 score: {}", r2(&dataset.target_vector, &pred));
//!
//! ```
//!


mod dataset;
mod split_criteria;
mod tests;
mod trees;
mod utils;

pub use dataset::Dataset;
pub use trees::DecisionTree;
pub use trees::RandomForest;
pub use trees::TrainOptions;
pub use trees::Tree;
pub use utils::{accuracy, r2};

use pyo3::prelude::*;

#[pymodule]
fn rustrees(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<DecisionTree>()?;
    m.add_class::<RandomForest>()?;
    Ok(())
}
