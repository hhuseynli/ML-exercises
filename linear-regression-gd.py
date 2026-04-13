import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, max_iter: int = 10000, alpha: float = 0.01, epsilon: float = 1e-8, w_init: np.ndarray | None = None, b_init: float = 0.0):
        self.max_iter = max_iter
        self.alpha = alpha
        self.w = w_init
        self.b = b_init
        self.cost_history = np.zeros(max_iter)
        self.epsilon = epsilon

    def fit(self, x, y):
        m, n = x.shape
        if self.w is None:
            self.w = np.zeros(n)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        x = self.normalize_features(x)    
        for iteration in range(self.max_iter):
            y_pred = self.predict(x, normalized=True)
            residuals = y_pred - y
            
            self.cost_history[iteration] = self.compute_cost_(residuals)
            delta = abs(self.cost_history[iteration - 1] - self.cost_history[iteration]) if iteration >= 1 else float('inf')
            if delta < self.epsilon:
                print(f"Convergence reached at iteration {iteration}.")
                self.cost_history = self.cost_history[:iteration + 1]
                break

            # Gradient updates
            dj_dw = np.dot(x.T, residuals) / m 
            dj_db = np.mean(residuals)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            
        return self.w, self.b
    
    def normalize_features(self, x):
        return (x - self.mean) / self.std
    
    def compute_cost_(self, residuals):
        return (residuals ** 2).mean() / 2
    
    def predict(self, x, normalized=False):
        if not normalized:
            x = self.normalize_features(x)
        return np.dot(x, self.w) + self.b
    
    def plot_learning_curve(self):
        if self.w is None:
            print("No cost history found. Fit the model first.")
            return
            
        plt.figure(figsize=(8, 5))
        # Use actual iteration counts for the X-axis
        iterations = np.arange(len(self.cost_history))
        plt.plot(iterations, self.cost_history, marker='o', color='#2ca02c')
        plt.title('Cost Reduction (Learning Curve)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def __repr__(self):
        if self.w is None:
            return "The model hasn't been fitted, run model.fit() first."
        weights = [f"{w_i:.4f}*x_{i}" for i, w_i in enumerate(self.w)]
        return "f(x) = " + " + ".join(weights) + f" + {self.b:.4f}"

if __name__ == "__main__":
    x = np.array([[1, 10000], [2, 30000], [3, 40000], [4, 50000]])
    y = np.array([9999, 29998, 39997, 49996])
    
    model = LinearRegressionGD(max_iter=10000, alpha=1, epsilon=1e-8)
    model.fit(x, y)
    
    print(f"Prediction for [5, 60000]: {model.predict(np.array([5, 60000])):.2f}")
    print(f"Model: {model}")
    model.plot_learning_curve()