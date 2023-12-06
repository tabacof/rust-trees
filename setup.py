from setuptools import find_packages, setup

setup(
    name="rustrees",
    version="0.2.0",
    description="Efficient decision tree and random forest library written in Rust with Python bindings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pedro Tabacof, Guilherme Lázari",
    author_email="tabacof@gmail.com, guilhermelcs@gmail.com",
    url="https://github.com/tabacof/rust-trees",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.0",
        "pandas>=2.0",
        "scikit-learn>=1.0",
        "pyarrow>=12.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Rust",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    setup_requires=["maturin>=1.0,<2.0"],
    zip_safe=False,
    project_urls={
        "Documentation": "https://rust-trees.readthedocs.io/en/latest/",
        "Rust Package": "https://crates.io/crates/rustrees",
    },
)
