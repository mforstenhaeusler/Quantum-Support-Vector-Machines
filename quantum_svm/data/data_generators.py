import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.datasets import ad_hoc_data

from quantum_svm.utils.utils import normalize

def create_bipolar_data(N, D, centers, sigma, seed, test_size=0.2):
    X, y = make_blobs(n_samples=N, n_features=D, centers=centers, cluster_std=sigma, random_state=seed)
    # normalize
    X = normalize(X)
    # make binary labels {-1, 1}
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(np.float), test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def create_XOR_data(N, D, centers, sigma, seed, test_size=0.2):
    X, y = make_blobs(n_samples=N, n_features=D, centers=centers, cluster_std=sigma, random_state=seed)
    # normalize
    X = normalize(X)

    # make y = {-1,1} where 0,3 = -1 and 1,2 are 1
    y_n = np.ones(len(y))
    for idx, elem in enumerate(y):
        #print(elem)
        if elem == 0 or elem == 3:
            y_n[idx] = -1
        else:
            y_n[idx] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y_n.astype(np.float), test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def create_circles_data(N, factor, noise, seed, test_size=0.2):
    X, y = make_circles(n_samples=N, factor=factor, noise=noise, random_state=seed)
    # normalize
    X = normalize(X)
    # make binary labels {-1, 1}
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(np.float), test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def create_moons_data(N, noise, seed, test_size=0.2):
    X, y = make_moons(n_samples=N, noise=noise, random_state=seed)
    # normalize
    X = normalize(X)
    # make binary labels {-1, 1}
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(np.float), test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def adhoc_dataset(train_size, test_size, adhoc_dimension, gap):
    # normalize
    #X = normalize(X)

    X_train, y_train, X_test, y_test, adhoc_total = ad_hoc_data(
        training_size=train_size,
        test_size= test_size, 
        n=adhoc_dimension,
        gap=gap,
        plot_data=False, one_hot=False, include_sample_total=True
    )
    # normalize
    #X_train, X_test = normalize(X_train), normalize(X_test)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    
    return X_train, y_train, X_test, y_test, adhoc_total



