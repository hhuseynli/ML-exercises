import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models.linear_regression_gd import LinearRegressionGD

x = np.array([[10, 20], [40, 50], [70, 80], [90, 110]])
y = np.array([410, 2540, 6470, 12190])
model = LinearRegressionGD(alpha=0.5, max_iter=100000, epsilon=1e-10)
poly = x[:, 1]**2
X = np.c_[x, poly]
w, b = model.fit(X, y)
print(f"Predicted value for [5, 10, 100]: {model.predict(np.array([5, 10, 100])):.2f}")
