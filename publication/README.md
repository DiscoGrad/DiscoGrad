# Reproduction

This folder contains everything needed to reconduct the empirical evaluation of the [publication](paper.pdf). The script `reproduce.py` will guide you through the process.

## Prerequisites

1. all prerequisites of the DiscoGrad tool, as documented in the [Readme](../README.md)
2. `Python 3.9` (works) or `Python 3.10` (needs one of the fixes below)
  - with the module Requirements listed in `requirements.txt`, which can be installed using `pip install -r requirements.txt`
  - Keep in mind that on some distributions you have to invoke Python 3 explicitly with `python3` (and `pip3`) and on others `python` (`pip`) is enough

- Notes on using Python 3.10
  - We rely on the `numpy_ml` library for gradient-based optimization, which is currently incompatible with Python 3.10, because [the Hashable class was moved](https://github.com/ddbourgin/numpy-ml/issues/79). There are two options to work around this:
    - Create a virtual environment, for example with `python -m venv .venv`, which will create a virtual environment in the folder `.venv`, change into it with `source .venv/bin/activate` and install the dependencies with `pip install -r requirements.txt`. Now navigate to the folder in the virtual environment where `numpy_ml` is installed and apply the fix as described in the link above.
    - Create a virtual environment with a tool that allows to choose the Python version, for example `virtualenv --python3.9 .venv`, which creates a virtual environment in the folder `.venv`, change into it with `source .venv/bin/activate` and install the dependencies with `pip install -r requirements.txt`

## Usage

Run `python reproduce.py` and follow the instructions.

