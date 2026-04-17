from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Simulate training data
x_train = np.arange(20)
y_train = 5 * x_train + 4

# Normalize input data
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train.reshape(-1, 1))

# Fit a linear model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, y_train)
print(f"Number of iterations: {sgdr.n_iter_}")

# View scaled parameters
print(sgdr.coef_, sgdr.intercept_)

y_pred = sgdr.predict(x_norm)

for pred in y_pred:
    print("{:.2f}".format(pred))
