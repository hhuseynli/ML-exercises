import numpy as np
import matplotlib.pyplot as plt
try:
    from .linear_regression_gd import LinearRegressionGD
except ImportError:
    from linear_regression_gd import LinearRegressionGD

class LogisticRegressionGD(LinearRegressionGD):
    def __init__(self, max_iter: int = 10000, alpha: float = 0.01, epsilon: float = 1e-8, w_init = None, b_init: float = 0.0):
        super().__init__(max_iter, alpha, epsilon, w_init, b_init)

    def sigmoid_(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss_(self, y_hat, y):
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, x, y):
        m, n = x.shape
        if self.w is None:
            self.w = np.zeros(n)
        
        x_norm = self.normalize_features(x, training=True)
         
        for iteration in range(self.max_iter):
            z = np.dot(x_norm, self.w) + self.b
            y_hat = self.sigmoid_(z)
            
            self.cost_history[iteration] = self.compute_loss_(y_hat, y)
            
            residuals = y_hat - y
            dj_dw = np.dot(x_norm.T, residuals) / m 
            dj_db = np.mean(residuals)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            
            if iteration > 0 and abs(self.cost_history[iteration-1] - self.cost_history[iteration]) < self.epsilon:
                self.cost_history = self.cost_history[:iteration + 1]
                break
            
        return self.w, self.b

    def predict(self, x):

        z = super().predict(x)
        return self.sigmoid_(z)

    def plot_decision_boundary(self, x, y):
        if x.shape[1] > 1:
            print("Visualization currently supported for 1D features only.")
            return

        plt.figure(figsize=(10, 6))
        
        # Data points
        plt.scatter(x, y, c=y, cmap='bwr', edgecolor='k', s=100, label='Data Points', zorder=5)
        
        # Sigmoid curve
        x_range = np.linspace(x.min() - 1, x.max() + 1, 300).reshape(-1, 1)
        y_probs = self.predict(x_range)
        plt.plot(x_range, y_probs, color='black', linewidth=2, label='Logistic Curve')
        
        # Boundary calculation (where z = 0)
        # x_norm = -b / w -> then de-normalize
        db_x_norm = -self.b / self.w[0]
        db_x = (db_x_norm * self.std[0]) + self.mean[0]

        # Boundary calculation (where z = 0)
        # x_norm = -b / w -> then de-normalize
        db_x_norm = -self.b / self.w[0]
        db_x = (db_x_norm * self.std[0]) + self.mean[0]
        
        plt.axvline(x=db_x, color='red', linestyle='--', label=f'Decision Boundary (x ≈ {db_x:.2f})')
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        
        plt.title("Logistic Regression: Decision Boundary & Sigmoid Fit")
        plt.xlabel("Feature X")
        plt.ylabel("Probability P(y=1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()