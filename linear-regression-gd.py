import numpy as np

class LinearRegressionGD:
    def __init__(self, max_iter: int = 10000, alpha: float = 0.01, w_init: np.ndarray | None = None, b_init: float = 0.0):
        self.max_iter = max_iter
        self.alpha = alpha
        self.w = w_init
        self.b = b_init

    def fit(self, x, y):
        m, n = x.shape
        
        if self.w is None:
            self.w = np.zeros(n)
            
        for _ in range(self.max_iter):
            # Calculate predictions
            y_pred = self.predict(x)
            residuals = y_pred - y
            
            # Compute gradients
            dj_dw = np.dot(x.T, residuals) / m # Vectorized dj_dw: (n, m) dot (m,) -> (n,)
            dj_db = np.mean(residuals)

            # Update parameters
            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            
        return self.w, self.b

    def predict(self, x):
        return np.dot(x, self.w) + self.b
    
    def __repr__(self):
        if self.w is None:
            return "The model hasn't been fitted, run model.fit() first."
        weights = [f"{w_i:.4f}*x_{i}" for i, w_i in enumerate(self.w)]
        return "f(x) = " + " + ".join(weights) + f" + {self.b:.4f}"

if __name__ == "__main__":
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([3, 5, 7, 9])
    
    model = LinearRegressionGD(max_iter=10000, alpha=0.04)
    model.fit(x, y)
    
    print(f"Model: {model}")
    print(f"Prediction for [5, 6]: {model.predict(np.array([5, 6]))}")