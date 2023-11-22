# Contributing to Rustrees

Thank you for your interest in contributing to Rustrees! This document outlines how to contribute to the project.

## Table of Contents

- [Contributing to Rustrees](#contributing-to-rustrees)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
  - [Development Setup](#development-setup)
    - [Python](#python)
    - [Rust](#rust)
  - [Submitting Changes](#submitting-changes)
  - [Testing](#testing)
    - [Python](#python-1)
    - [Rust](#rust-1)
    - [Testing Guidelines](#testing-guidelines)
  - [Documentation](#documentation)
  - [Issue Tracking](#issue-tracking)
  - [Release Process](#release-process)

## Code of Conduct

All contributors must agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to [email@rustrees.com](mailto:email@rustrees.com).

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Create a new branch for your work.

## Development Setup

### Python

```bash
pip install -r requirements_dev.txt
```

### Rust

```bash
cargo build
```

## Submitting Changes

1. Make sure your code passes all tests.
2. Rebase your branch to make sure you have the latest changes from `main`.
3. Push your branch to your fork on GitHub.
4. Submit a pull request from your fork to the Rustrees `main` branch.

## Testing

### Python

```bash
pytest
```

### Rust

```bash
cargo test
```

### Testing Guidelines

- Write comprehensive and thorough tests.
- All tests must pass before a pull request can be merged.

## Documentation

- Update the README.md if you introduce new features or changes.
- For Rust, follow the rustdoc format. For Python, use docstrings in the numpydoc format.

## Issue Tracking

Please use the GitHub issue tracker to report issues or suggest features.

## Release Process

Releases are cut from the `main` branch, and release candidates should have their versions bumped according to [Semantic Versioning](https://semver.org/).

---

Feel free to tailor this to your project's specific needs.