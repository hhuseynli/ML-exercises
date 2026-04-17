import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models.linear_regression_gd import LinearRegressionGD

x = np.array([[10, 20, 50], [40, 50, 100], [70, 60, 150], [40, 30, 120], [90, 110, 200]])
y = np.array([250, 2100, 4350, 1320, 10100])
model = LinearRegressionGD(alpha=0.1, max_iter=100000, epsilon=1e-10)
new_feature = x[:, 0] * x[:, 1]
X = np.c_[x, new_feature]
w, b = model.fit(X, y)
print(f"Predicted value for [5, 10, 100]: {model.predict(np.array([5, 10, 100, 50])):.2f}")
