import numpy as np
import functools

def hadamard_gate():
    H = np.ones((2,2), dtype=np.complex_)
    H[1,1] = -1.
    return 1/np.sqrt(2)*H

def X_gate():
    X = np.zeros((2,2), dtype=np.complex_)
    X[0,1] = 1.
    X[1,0] = 1.
    return X

def Y_gate():
    X = np.array([[0, -1j], [1j, 0]], dtype=np.complex_)
    return X

def Z_gate():
    X = np.eye(2,2)
    X[1,1] = -1.
    return X

def CNOT(gate):
    U = np.eye(4,dtype=np.complex_)
    U[2:, 2:] = gate
    return U 

def data_map_func(x) -> float:
    """ Data map function, f: R^n -> R
    
    Params:
    -------
    x : np.ndarray
        data

    Returns:
    --------
    coef : float
           the mapped value
    """
    coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: m * n, np.pi - x)
    return coeff

def data_map_func_sin(x) -> float:
    """ Data map function, f: R^n -> R
    
    Params:
    -------
    x : np.ndarray
        data

    Returns:
    --------
    coef : float
           the mapped value
    """
    coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: m * n, np.sin(np.pi - x))
    return coeff

def data_map_func_sin(x) -> float:
    """ Data map function, f: R^n -> R
    
    Params:
    -------
    x : np.ndarray
        data

    Returns:
    --------
    coef : float
           the mapped value
    """
    coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: np.pi*np.exp(((m - n)*(m - n))/8), x)
    return coeff

    