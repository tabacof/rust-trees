# Rustrees: Decision Trees & Random Forests in Rust with Python Bindings

[![Build Status](https://travis-ci.com/yourusername/rustrees.svg?branch=main)](https://travis-ci.com/yourusername/rustrees)
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

### Limitations

- ⚙️ **Limited Configuration**: Currently supports a basic set of hyperparameters.

## Python

### Installation

```bash
pip install rustrees
```

### Quick Start

```python
from rustrees import DecisionTreeClassifier

X = [[1, 2], [3, 4], [5, 6]]
y = [0, 1, 0]

clf = DecisionTreeClassifier()
clf.fit(X, y)

predictions = clf.predict([[2, 3], [4, 5]])
```

## Rust

### Installation

```bash
cargo install rustrees
```

### Quick Start

```rust
use rustrees::DecisionTreeClassifier;

let X = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
let y = vec![0, 1, 0];

let mut clf = DecisionTreeClassifier::new();
clf.fit(&X, &y);

let predictions = clf.predict(&vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
```

### Development 

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
from rustrees.rustrees import DecisionTree, RandomForest
import rustrees.tree as rt
```

## API Documentation

- Python API: [Link](https://your-python-docs-link.com)
- Rust API: [Link](https://docs.rs/rustrees)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
