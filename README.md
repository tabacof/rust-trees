# Rustrees: Decision Trees & Random Forests in Rust with Python Bindings

[![Build Status](https://github.com/tabacof/rust-trees/actions/workflows/rust.yml/badge.svg)](https://github.com/tabacof/rust-trees/actions)
[![PyPI version](https://badge.fury.io/py/rustrees.svg)](https://badge.fury.io/py/rustrees)
[![Rust Package](https://img.shields.io/crates/v/rustrees)](https://crates.io/crates/rustrees)
[![Documentation](https://docs.rs/rustrees/badge.svg)](https://docs.rs/rustrees)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

Rustrees is an efficient decision tree and random forest library written in Rust with Python bindings. It aims to provide speed comparable to Sklearn with the reliability and performance of Rust.

### Features

- 🏎️ **Speed**: As fast as Sklearn on average.
- 🔗 **Python Bindings**: Effortless integration with Python.
- 🔒 **Type Safety**: Benefit from Rust's strong type system.

## Python

### Installation

```bash
pip install rustrees
```

### Quick Start

```python
from sklearn.metrics import accuracy_score
from sklearn import datasets
import rustrees.decision_tree as rt_dt

df = datasets.load_breast_cancer()

model = rt_dt.DecisionTreeClassifier(max_depth=5).fit(df["data"], df["target"])

acc = accuracy_score(df["target"], model.predict(df["data"]))

print("accuracy", acc)
```

## Rust

### Installation

```bash
cargo add rustrees
```

### Quick Start

```rust
use rustrees::{DecisionTree, Dataset, r2};

let dataset = Dataset::read_csv("iris.csv", ",");

let dt = DecisionTree::train_reg(
   &dataset, 
   5,        // max_depth
   Some(1),  // min_samples_leaf        
   Some(42), // random_state
);

let pred = dt.predict(&dataset);

println!("r2 score: {}", r2(&dataset.target_vector, &pred));
```

## Developing 

First, create a virtualenv (this just needs to be done once):
```bash
python -m venv .env
```

Then, activate the virtualenv (needs to be done every time):
```bash
source .env/bin/activate
```

Now, install the requirements (just needs to be done once):
```bash
pip install -r requirements.txt
```

Finally, install the Python library at the local virtual environment with the following command (needs to be done every time you change the Rust code):
```bash
maturin develop --release
```

Now, you can import the library `rustrees` in Python. This can be done also from Jupyter notebooks. To do so, run the following command:
```bash
jupyter notebook
```

And then import the library in the notebook:
```python
import rustrees.decision_tree as rt_dt
```
