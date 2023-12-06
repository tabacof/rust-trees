import toml

# Load and parse Cargo.toml
with open("path/to/your/Cargo.toml", "r") as file:
    cargo_toml = toml.load(file)

# Load and parse pyproject.toml
with open("path/to/your/pyproject.toml", "r") as file:
    pyproject_toml = toml.load(file)

# Extract information from Cargo.toml
package_name = cargo_toml["package"]["name"]
version = cargo_toml["package"]["version"]
description = cargo_toml["package"].get("description", "")
license_type = cargo_toml["package"].get("license", "")
authors = cargo_toml["package"].get("authors", [])
authors_str = ", ".join(authors)

# Extract information from pyproject.toml
dependencies = pyproject_toml["project"]["dependencies"]
build_requires = pyproject_toml["build-system"]["requires"]

from setuptools import find_packages, setup

setup(
    name=package_name,
    version=version,
    description=description,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author=authors_str,
    url='https://github.com/tabacof/rust-trees',
    license=license_type,
    packages=find_packages(),
    include_package_data=True,
    install_requires=dependencies,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Rust',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    setup_requires=build_requires,
    zip_safe=False,
    project_urls={
        'Documentation': 'https://rust-trees.readthedocs.io/en/latest/',
        'Rust Package': 'https://crates.io/crates/rustrees',
    },
)
