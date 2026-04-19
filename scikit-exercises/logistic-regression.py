import numpy as np
from sklearn.linear_model import LogisticRegression

# Simulate training data
x_train = np.arange(8).reshape(-1, 1)
y_train = np.concatenate([np.zeros(4), np.ones(4)])

# Fit the model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

# Predict outputs
print(lr_model.predict(x_train))

# Output accuracy
print(lr_model.score(x_train, y_train))