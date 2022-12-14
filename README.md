# Neural Network for Digital Twin (NNDT)

[![GitHub issues by-label](https://img.shields.io/github/issues/nndt-team/nndt/good%20first%20issue)](https://github.com/nndt-team/nndt/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
[![docstr_coverage](docstr-cov-badge.svg)](https://github.com/nndt-team/nndt)
[![codecov](https://codecov.io/gh/nndt-team/nndt/branch/main/graph/badge.svg)](https://codecov.io/gh/nndt-team/nndt)
[![CodeFactor](https://www.codefactor.io/repository/github/nndt-team/nndt/badge)](https://www.codefactor.io/repository/github/nndt-team/nndt)
[![ci-build workflow](https://github.com/nndt-team/nndt/actions/workflows/ci-build.yml/badge.svg)](https://github.com/nndt-team/nndt/actions/workflows/ci-build.yml)
[![Documentation Status](https://readthedocs.org/projects/nndt/badge/?version=latest)](https://nndt.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/nndt-team/nndt)](https://github.com/nndt-team/nndt/blob/main/LICENSE)
[![Python version](python-badge.svg)](https://www.python.org/downloads/release/python-3110/)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/nndt-team/nndt)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[Wiki](wiki) | [Change logs](CHANGELOG.md)

This is experimental library for unusual neural network applications. Description and docs will come soon!

## Install
 
The last stable release:
```
pip install git+https://github.com/nndt-team/nndt.git
```

The developer release:
```
pip install git+https://github.com/nndt-team/nndt.git@dev
```

[Installation for Windows with WSL](wiki/NNDT-WSL)

## Test

For the test runs, configure your IDE to run test from `test` folder and assume `nndt/nndt` as a source root. 
Another option is to copy `test` into `nndt` folder and run all together.
 
## Tutorials and examples

NOTE! This project is in an early stage. API and examples are changed every day. 

Tutorials
1. [Basics for space models](tutorials/tutorial1_space_model.ipynb)
2. [Data visualization](tutorials/tutorial2_research_viz.ipynb)

Examples for conference papers:
1. [SDT2SDF conversion for the one object](examples/sdt2sdf_for_one_object.py)
2. [Shape regression for two objects](examples/shape_regression.py)

Tests for manual evaluation
1. [Shape interpolation](tests_manual/sdf_multiple_files.py)
2. [Mesh segmentation (supervised)](tests_manual/mesh_segmentation.py)
3. [Eikonal equation solution (geodesic distance in 3D)](tests_manual/eikonal_on_primitives.py)

