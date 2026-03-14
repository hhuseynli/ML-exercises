# ML-exercises

Hands-on implementations of machine learning concepts from scratch, for a deep practical understanding.

## Motivation

Understanding an algorithm conceptually is not the same as understanding it practically. This repository goes beyond textbook explanations: every concept is coded up from the ground up, without relying on high-level library abstractions, so that the underlying mathematics and mechanics are fully transparent.

For example, instead of calling `numpy.linalg.lstsq`, the least-squares method is implemented by hand — deriving the normal equations, solving the linear system, and verifying the result — to build genuine intuition about what the algorithm actually does.

## Exercises

| # | Concept | Script |
|---|---------|--------|
| 1 | Least-Squares Method | `least_squares.py` |

## How to Run

```bash
python least_squares.py
```

Each script is self-contained and prints its results to the console so you can inspect every intermediate step.

## Structure

```
ML-exercises/
├── README.md
└── least_squares.py   # Normal-equation implementation of linear least squares
```
