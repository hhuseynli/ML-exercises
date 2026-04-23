import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models.linear_regression_gd import LinearRegressionGD
import matplotlib.pyplot as plt

# Pattern used in this toy dataset: y = x1 + x2^2
x = np.array([
	[10, 20],
	[20, 30],
	[30, 40],
	[40, 50],
	[50, 60],
	[60, 70],
	[70, 80],
	[80, 90],
	[90, 100],
	[100, 110],
], dtype=float)
y = x[:, 0] + x[:, 1] ** 2

poly_model = LinearRegressionGD(alpha=0.5, max_iter=100000, epsilon=1e-10)
poly = x[:, 1] ** 2
X_poly = np.c_[x, poly]
poly_model.fit(X_poly, y)

# Baseline model without polynomial feature (plain linear regression on x1, x2).
linear_model = LinearRegressionGD(alpha=0.5, max_iter=100000, epsilon=1e-10)
linear_model.fit(x, y)

# Build prediction inputs with the same feature engineering used for training.
sample = np.array([5, 10])
sample_poly = sample[1] ** 2
sample_features = np.array([sample[0], sample[1], sample_poly])
true_value = sample[0] + sample_poly
poly_pred = poly_model.predict(sample_features)
linear_pred = linear_model.predict(sample)

print(f"True value for [5, 10, 100]: {true_value:.2f}")
print(f"Polynomial model prediction: {poly_pred:.2f}")
print(f"Linear model prediction: {linear_pred:.2f}")

poly_train_pred = poly_model.predict(X_poly)
linear_train_pred = linear_model.predict(x)
poly_mse = np.mean((poly_train_pred - y) ** 2)
linear_mse = np.mean((linear_train_pred - y) ** 2)
print(f"Train MSE (Polynomial): {poly_mse:.6f}")
print(f"Train MSE (Linear): {linear_mse:.6f}")

# Plot learning curve
poly_model.plot_learning_curve()
plt.scatter(x[:, 0], y, color='blue', label='Data Points')
x_plot = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
x2_plot = np.interp(x_plot, x[:, 0], x[:, 1])
X_plot = np.column_stack([x_plot, x2_plot, x2_plot**2])
y_poly_plot = poly_model.predict(X_plot)
y_linear_plot = linear_model.predict(np.column_stack([x_plot, x2_plot]))
plt.plot(x_plot, y_poly_plot, color='red', label='Polynomial Model')
plt.plot(x_plot, y_linear_plot, color='orange', linestyle='--', label='Linear Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
