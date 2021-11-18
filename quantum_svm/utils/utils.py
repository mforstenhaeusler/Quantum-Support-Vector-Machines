import numpy as np


def normalize(X):
    """ Normalizes Data """
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    
    return (X - x_min)/(x_max - x_min)

def accuracy(y, y_pred, mode):
    y_true = y[y == y_pred]
    print(f'Accuracy on {mode} set: {len(y_true)/len(y) * 100} %')