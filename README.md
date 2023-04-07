# Rustrees
Decision trees, random and causal forests in Rust.

Work in progress, stay tuned!

## Using Python library

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
import rustrees
```
