import numpy as np

             
def linear_kernel(x1, x2):
    """ Linear Kernel (refer to slide 26) """
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, c=1):
    """ Polynominal Kernel (refer to slide 26) """
    #return (1 + gamma*np.dot(x, y)) ** degree
    return (c + np.dot(x1, x2)) ** degree

def gaussian_kernel(x1, x2, gamma=0.5):
    """ Gaussian Kernel (refer to slide 26) """
    return np.exp( -gamma * np.linalg.norm(x1 - x2)**2 )

def rbf_kernel(x1, x2, gamma=0.5):
    """ Radial Basis Function Kernel (refer to slide 26)
    gamma = 1/((2*sigma)**2) """
    return np.exp( -gamma * np.dot(x1 - x2, x1 - x2) )

def sigmoid_kernel(x1, x2, gamma=0.5, c=1):
    """ Sigmoid Kernel (refer to slide 26) """
    return np.tanh(gamma * np.dot(x1, x2) + c)