import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, max_iter: int = 10000, alpha: float = 0.01, epsilon: float = 1e-8, w_init = None, b_init: float = 0.0):
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
        
        x_norm = self.normalize_features(x, training=True)    
        
        for iteration in range(self.max_iter):
            y_pred = np.dot(x_norm, self.w) + self.b
            residuals = y_pred - y
            
            self.cost_history[iteration] = self.compute_cost_(residuals)
            
            # Gradient updates
            dj_dw = np.dot(x_norm.T, residuals) / m 
            dj_db = np.mean(residuals)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            
            # Check convergence
            if iteration > 0 and abs(self.cost_history[iteration-1] - self.cost_history[iteration]) < self.epsilon:
                self.cost_history = self.cost_history[:iteration + 1]
                break
            
        return self.w, self.b
    
    def predict(self, x):
        x_norm = self.normalize_features(x)
        return np.dot(x_norm, self.w) + self.b
    
    def normalize_features(self, x, training=False):
        if training:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        return (x - self.mean) / self.std
    
    def compute_cost_(self, residuals):
        return (residuals ** 2).mean() / 2
    
    def plot_learning_curve(self):
        if self.w is None:
            print("No cost history found. Fit the model first.")
            return
            
        plt.figure(figsize=(8, 5))
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
    
        w_orig = self.w / self.std
        b_orig = self.b - np.sum(self.w * self.mean / self.std)
    
        weights = [f"{w_i:.4f}*x_{i}" for i, w_i in enumerate(w_orig)]
        return "f(x) = " + " + ".join(weights) + f" + {b_orig:.4f}"
