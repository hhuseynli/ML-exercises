import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from models.deep_neural_networks import DeepNeuralNetwork
from one_hot import encode, decode

lib = np.load("./data/MNIST.npz")

X_train_3D = lib["X_train"]
Y_train_raw = lib["Y_train"]
X_valid_3D = lib["X_valid"]
Y_valid_raw = lib["Y_valid"]


# Transform matrices to (features, examples) structure
X_train = X_train_3D.reshape(X_train_3D.shape[0], -1).T
X_valid = X_valid_3D.reshape(X_valid_3D.shape[0], -1).T

# Convert source single-column class targets to one-hot encoded vectors
Y_train = encode(Y_train_raw, 10).T
Y_valid = encode(Y_valid_raw, 10).T


model = DeepNeuralNetwork(X_train.shape[0], [256, 128, 64, 10])

X_train_subset = X_train[:, :10000]
Y_train_subset = Y_train[:, :10000]
Y_train_subset_raw = Y_train_raw[:10000]

A_train_onehot, train_cost = model.train(
    X_train_subset, Y_train_subset, 
    iterations=5000, alpha=0.3, 
    verbose=True, graph=True, step=200
)

# Transpose A to (examples, classes) to run your default decode function cleanly
A_train_decoded = decode(A_train_onehot.T)
train_accuracy = np.mean(A_train_decoded == Y_train_subset_raw) * 100

print("\n" + "="*30)
print(f"Final Train Cost: {train_cost}")
print(f"Train Accuracy:   {train_accuracy:.2f}%")
print("="*30 + "\n")


A_valid_onehot, valid_cost = model.evaluate(X_valid, Y_valid)

# Transpose A to (examples, classes) to run your default decode function cleanly
A_valid_decoded = decode(A_valid_onehot.T)
valid_accuracy = np.mean(A_valid_decoded == Y_valid_raw) * 100

print("="*30)
print(f"Validation Cost:  {valid_cost}")
print(f"Validation Accuracy: {valid_accuracy:.2f}%")
print("="*30 + "\n")

fig = plt.figure(figsize=(12, 12))

for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1)
    
    # Render the input 3D digit arrays as grayscale images
    plt.imshow(X_valid_3D[i], cmap='gray')
    
    # Assign the decoded network forecast integer directly as the title
    ax.set_title(int(A_valid_decoded[i]), fontsize=10, pad=2)
    
    # Clean visual grid artifacts from canvas subplots
    plt.axis('off')

plt.tight_layout()
plt.show()