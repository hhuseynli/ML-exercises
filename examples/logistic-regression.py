import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.logistic_regression_gd import LogisticRegressionGD
import numpy as np

x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Initialize and train
model = LogisticRegressionGD(alpha=0.1, max_iter=2000)
model.fit(x_train, y_train)

# Output predictions
probs = model.predict(x_train)
print("Predictions (Probabilities):")
print(probs)

# Plot results
model.plot_learning_curve()
model.plot_decision_boundary(x_train, y_train)