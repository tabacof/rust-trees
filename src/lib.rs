mod dataset;
mod split_criteria;
mod tests;
mod trees;
mod utils;

pub use dataset::Dataset;
pub use trees::DecisionTree;
pub use utils::*;

use pyo3::prelude::*;

#[pymodule]
fn rustrees(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<DecisionTree>()?;
    Ok(())
}
