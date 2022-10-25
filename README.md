# Neural Network for Digital Twin (NNDT)

[![GitHub issues by-label](https://img.shields.io/github/issues/KonstantinUshenin/nndt/good%20first%20issue)](https://github.com/KonstantinUshenin/nndt/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
[![docstr_coverage](https://github.com/KonstantinUshenin/nndt/blob/main/badge.svg)](https://github.com/KonstantinUshenin/nndt)
[![codecov](https://codecov.io/gh/KonstantinUshenin/nndt/branch/main/graph/badge.svg)](https://codecov.io/gh/KonstantinUshenin/nndt)
[![CodeFactor](https://www.codefactor.io/repository/github/konstantinushenin/nndt/badge)](https://www.codefactor.io/repository/github/konstantinushenin/nndt)
[![ci-test workflow](https://github.com/KonstantinUshenin/nndt/actions/workflows/ci-test.yml/badge.svg)](https://github.com/KonstantinUshenin/nndt/actions/workflows/ci-test.yml)
[![License](https://img.shields.io/github/license/KonstantinUshenin/nndt)](https://github.com/KonstantinUshenin/nndt/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/KonstantinUshenin/nndt)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[Wiki](https://github.com/KonstantinUshenin/nndt/wiki) | [Change logs](https://github.com/KonstantinUshenin/nndt/blob/v0.0.1rc/CHANGELOG.md)

This is experimental library for unusual neural network applications. Description and docs will come soon!

## Install
 
The last stable release:
```
pip install git+https://github.com/KonstantinUshenin/nndt/releases/tag/v0.0.2
```

The developer release:
```
pip install git+https://github.com/KonstantinUshenin/nndt.git
```

Installation for Windows with WSL:
```
https://github.com/KonstantinUshenin/nndt/wiki/NNDT-WSL
```

## Test

How to run the tests?
 - Unpack `./test/acdc_for_test.7z`.
 - Configure your IDE to run test from `test` folder and assume `nndt/nndt` as a source root.
 - Run test;)
 
## Tutorials and examples

NOTE! This project is in an early stage. API and examples are changed every day. 
The following notebooks use `v0.0.1rc` branch.

Tutorials
1. [Space model and generators](https://drive.google.com/file/d/16VEUCfcCtRQOYGqe6N2MBsIsD8OILufL/view?usp=sharing)
2. [Trainable tasks](https://drive.google.com/file/d/16ZnfqzL9VsGqnyWG4zV9uVcwFSvlHdYN/view?usp=sharing)

Examples
1. [Shape interpolation](https://github.com/KonstantinUshenin/nndt/blob/main/demos_preliminary/sdf_multiple_files.py)
2. [Mesh segmentation (supervised)](https://github.com/KonstantinUshenin/nndt/blob/main/demos_preliminary/mesh_segmentation.py)
3. [Eikonal equation solution (geodesic distance in 3D)](https://github.com/KonstantinUshenin/nndt/blob/main/demos_preliminary/eikonal_on_primitives.py)

