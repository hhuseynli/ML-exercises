# ML-exercises

Hands-on implementations of machine learning concepts from scratch, for a deep practical understanding.

## Motivation

Understanding an algorithm conceptually is not the same as understanding it practically. This repository goes beyond textbook explanations: every concept is coded up from the ground up, without relying on high-level library abstractions, so that the underlying mathematics and mechanics are fully transparent.

For example, instead of calling `numpy.linalg.lstsq`, the least-squares method is implemented by hand — deriving the normal equations, solving the linear system, and verifying the result — to build genuine intuition about what the algorithm actually does.

## Project Structure

```
models/      — Class implementations of ML algorithms from scratch
examples/    — Practical applications using those implementations
```

## Exercises

Based on Andrew Ng's Machine Learning Specialization.

### Course 1: Supervised Machine Learning

| # | Concept | Type | Script |
|---|---------|------|--------|
| 1 | Linear Regression (Gradient Descent) | model | `models/linear_regression_gd.py` |
| 2 | Linear Regression | example | `examples/linear-regression.py` |
| 3 | Polynomial Regression | example | `examples/polynomial-regression.py` |
| 4 | Feature Engineering | example | `examples/feature-engineering.py` |

### Planned

| # | Concept | Type |
|---|---------|------|
| 5 | Logistic Regression | model |
| 6 | Overfitting & Regularization | example |
| 7 | Least-Squares Method (Normal Equation) | model |

## How to Run

```bash
python3 models/linear_regression_gd.py
python3 examples/feature-engineering.py
```

Each script is self-contained and prints its results to the console so you can inspect every intermediate step.
