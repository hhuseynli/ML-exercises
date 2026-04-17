import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models.linear_regression_gd import LinearRegressionGD

x = np.array([[1, 10000], [2, 30000], [3, 40000], [4, 50000]])
y = np.array([9999, 29998, 39997, 49996])

model = LinearRegressionGD(max_iter=10000, alpha=1, epsilon=1e-12)
model.fit(x, y)

print(f"Prediction for [6, 10]: {model.predict(np.array([6, 10])):.2f}")
print(f"Model: {model}")
model.plot_learning_curve()
