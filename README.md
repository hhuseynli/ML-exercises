# ML-exercises

Hands-on implementations of machine learning concepts from scratch, for a deep practical understanding.

## Motivation

Understanding an algorithm conceptually is not the same as understanding it practically. This repository goes beyond textbook explanations: every concept is coded up from the ground up, without relying on high-level library abstractions, so that the underlying mathematics and mechanics are fully transparent.

For example, instead of calling `numpy.linalg.lstsq`, the least-squares method is implemented by hand — deriving the normal equations, solving the linear system, and verifying the result — to build genuine intuition about what the algorithm actually does.

## Exercises

Based on Andrew Ng's Machine Learning Specialization.

### Course 1: Supervised Machine Learning

| # | Concept | Script |
|---|---------|--------|
| 1 | Gradient Descent for Linear Regression | `linear-regression-gd.py` |
| 2 | Least-Squares Method (Normal Equation) | `least-squares.py` |
| 3 | Feature Engineering & Scaling | `feature-scaling.py` |
| 4 | Logistic Regression | `logistic-regression.py` |
| 5 | Overfitting & Regularization | `regularization.py` |

### Course 2: Advanced Learning Algorithms

| # | Concept | Script |
|---|---------|--------|
| 6 | Neural Network from Scratch | `neural-network.py` |
| 7 | Multiclass Classification (Softmax) | `softmax-regression.py` |
| 8 | Decision Tree | `decision-tree.py` |
| 9 | Random Forest & Boosting | `ensemble-methods.py` |
| 10 | Bias-Variance & Model Selection | `bias-variance.py` |

### Course 3: Unsupervised Learning & Recommenders

| # | Concept | Script |
|---|---------|--------|
| 11 | K-Means Clustering | `kmeans.py` |
| 12 | Anomaly Detection | `anomaly-detection.py` |
| 13 | Collaborative Filtering | `collaborative-filtering.py` |
| 14 | Principal Component Analysis | `pca.py` |
| 15 | Reinforcement Learning (Q-Learning) | `reinforcement-learning.py` |

## How to Run

```bash
python linear-regression-gd.py
```

Each script is self-contained and prints its results to the console so you can inspect every intermediate step.
