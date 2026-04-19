# ML-exercises

Hands-on implementations of machine learning concepts from scratch, for a deep practical understanding.

## Motivation

Understanding an algorithm conceptually is not the same as understanding it practically. This repository goes beyond textbook explanations: every concept is coded up from the ground up, without relying on high-level library abstractions, so that the underlying mathematics and mechanics are fully transparent.

For example, instead of calling `numpy.linalg.lstsq`, the least-squares method is implemented by hand — deriving the normal equations, solving the linear system, and verifying the result — to build genuine intuition about what the algorithm actually does.

## Project Structure

```
models/            — ML algorithm classes implemented from scratch
examples/          — End-to-end examples using custom models
scikit-exercises/  — Equivalent exercises with scikit-learn
```

## Exercises

Based on Andrew Ng's Machine Learning Specialization.

### Course 1: Supervised Machine Learning

| # | Concept | Type | Script |
|---|---------|------|--------|
| 1 | Linear Regression (Gradient Descent) | model | `models/linear_regression_gd.py` |
| 2 | Logistic Regression (Gradient Descent) | model | `models/logistic_regression_gd.py` |
| 3 | Linear Regression | example | `examples/linear-regression.py` |
| 4 | Polynomial Regression | example | `examples/polynomial-regression.py` |
| 5 | Feature Engineering | example | `examples/feature-engineering.py` |
| 6 | Logistic Regression | example | `examples/logistic-regression.py` |
| 7 | Linear Regression with `SGDRegressor` | scikit exercise | `scikit-exercises/linear-regression.py` |
| 8 | Logistic Regression with `LogisticRegression` | scikit exercise | `scikit-exercises/logistic-regression.py` |

## How to Run

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run from-scratch examples:

```bash
python3 examples/linear-regression.py
python3 examples/polynomial-regression.py
python3 examples/feature-engineering.py
python3 examples/logistic-regression.py
```

Run scikit-learn versions:

```bash
python3 scikit-exercises/linear-regression.py
python3 scikit-exercises/logistic-regression.py
```

## Notes

- Example scripts print predictions to the console.
- Some scripts also display Matplotlib plots (learning curve and logistic decision boundary).
- `examples/logistic-regression.py` currently visualizes decision boundaries for 1D features.
