import numpy as np


def encode(Y, classes):
    identity = np.eye(classes)
    return identity[Y]

def decode(Y):
    return np.argmax(Y, axis=1)