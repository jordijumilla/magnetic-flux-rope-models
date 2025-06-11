# Magnetic Flux Rope Models

Welcome to the **Magnetic Flux Rope Models** library — a Python package for modeling and analyzing magnetic flux rope structures using different analytical and numerical models. This library is designed for researchers, engineers, and data scientists working in space physics, solar-terrestrial relations, or any field where magnetic flux ropes are studied.

## 🌌 Overview

This package provides implementations of multiple magnetic flux rope models, including:

- **Elliptical Cylindrical Model (ECModel)**
- **Lundquist Model**
- **Circular Cross-section Model (CCModel)**

Each model builds upon a common base class (`MFRBaseModel`) and offers utilities for computing field components, fitting parameters to observed data, and generating synthetic events.

## 📦 Features

- Modular and extensible structure
- Support for elliptical coordinate systems
- Optimisation engine for parameter fitting
- Tools for adding and handling synthetic noise
- Test suite with high code coverage
- Compatible with Python 3.10+

## 📖 Documentation

To get started, check out the [Getting Started](documentation.md) section for installation and usage examples.

## 📐 Math and Physics

The mathematical expressions used in this library can be viewed directly in the docs, thanks to full support for $\LaTeX$ equations via MathJax.

## 📊 Testing & Coverage

Tests are run using `pytest` with coverage reporting enabled.

[![Tests](https://github.com/jordi-jumilla-lorenz/magnetic-flux-rope-models/actions/workflows/tests.yml/badge.svg)](https://github.com/jordi-jumilla-lorenz/magnetic-flux-rope-models/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/jordi-jumilla-lorenz/magnetic-flux-rope-models)](https://codecov.io/gh/jordi-jumilla-lorenz/magnetic-flux-rope-models)

## 🛠 Installation

```bash
pip install magnetic-flux-rope-models
