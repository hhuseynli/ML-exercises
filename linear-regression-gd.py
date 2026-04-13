import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, max_iter: int = 10000, alpha: float = 0.01, w_init: np.ndarray | None = None, b_init: float = 0.0):
        self.max_iter = max_iter
        self.alpha = alpha
        self.w = w_init
        self.b = b_init
        # Ensure threshold is an integer
        self.threshold = max(1, self.max_iter // 10) 
        # Calculate size to avoid off-by-one errors
        history_size = int(np.ceil(self.max_iter / self.threshold))
        self.cost_history = np.zeros(history_size)

    def fit(self, x, y):
        m, n = x.shape
        if self.w is None:
            self.w = np.zeros(n)
            
        for iteration in range(self.max_iter):
            y_pred = self.predict(x)
            residuals = y_pred - y
            
            if iteration % self.threshold == 0:
                index = iteration // self.threshold
                if index < len(self.cost_history):
                    self.cost_history[index] = self.compute_cost_(residuals)

            # Gradient updates
            dj_dw = np.dot(x.T, residuals) / m 
            dj_db = np.mean(residuals)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            
        return self.w, self.b
    
    def compute_cost_(self, residuals):
        return (residuals ** 2).mean() / 2
    
    def predict(self, x):
        return np.dot(x, self.w) + self.b
    
    def plot_learning_curve(self):
        if self.w is None:
            print("No cost history found. Fit the model first.")
            return
            
        plt.figure(figsize=(8, 5))
        # Use actual iteration counts for the X-axis
        iterations = np.arange(len(self.cost_history)) * self.threshold
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
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([3, 5, 7, 9])
    
    model = LinearRegressionGD(max_iter=10, alpha=0.01)
    model.fit(x, y)
    
    print(f"Prediction for [5, 6]: {model.predict(np.array([5, 6]))}")
    print(f"Model: {model}")
    model.plot_learning_curve()