from re import VERBOSE
import numpy as np 
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly

from .classical_kernels import linear_kernel, gaussian_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
from quantum_svm.quantum.quantum_kernels import quantum_kernel 
from quantum_svm.utils.utils import accuracy
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from numba import jit

class kernelSVC:
    def __init__(self, quantum_parans, kernel='linear', C=None, gamma=0.5, degree=3, alpha_tol=1e-4, verbose=True):
        """ Support Vector Classifier implementing the Kernel Trick on SVM dual problem
        Parameters
        ----------
        kernel : str
                 indicates the kernel in use
                
        C : float
            determines the margin strength
        
        gamma : float
                gamma = 1/((2*sigma) ** 2)
        
        degree : int
                 degree of polynominal kernel 

        alpha_tol : float,
                    minimum alpha value
        
        quantum_params : dict
                         dictionary of quantum parameters required for computation
        
        verbose : bool
                  determines if report is printed

        Returns
        -------
        params : dict
                 Dictionary with all the relevant parameters for display the classifiation.
        """
        self.kernel = kernel 
        if self.kernel == 'linear' or self.kernel == 'polynominal' or self.kernel == 'gaussian' or self.kernel == 'rbf' or self.kernel == 'sigmoid' or self.kernel == 'quantum':
            pass
        else:
            raise KernelError
        self.quantum_params = quantum_parans
        self.C = C
        if C is not None: self.C = float(self.C)
        self.gamma = gamma
        self.degree = degree 
        self.alphas = None
        self.alpha_tol = alpha_tol
        self.verbose = verbose
            
        self.gramMatrix = 0

        self.params = {}

        self.alphas = None
        self.sv_X = None
        self.sv_y = None
        self.w = None
        self.b = None
        self.is_fit = False

        self.qk = quantum_kernel(self.quantum_params)

        self.decision_function = None
        
        if self.C is None:
            print(f"SVC(kernel='{self.kernel}')")
        elif self.kernel == 'polynominal': 
            print(f"SVC(kernel='{self.kernel}', C={self.C}, degree={self.degree})")
        elif self.kernel == 'gaussian' or self.kernel == 'rbf' or self.kernel == 'sigmoid': 
            print(f"SVC(kernel='{self.kernel}', C={self.C}, gamma={self.gamma})")
        else:
            print(f"SVC(kernel='{self.kernel}', C={self.C})")
    
    #@jit
    def fit(self, X, y):
        """ Solves SVM dual problem with a QP solver.

        Parameters
        ----------
        X : array, shape [N, D]
            Input features.
        y : array, shape [N]
            Binary class labels (in {-1, 1} format).

        Returns
        -------
        alphas : array, shape [N]
                 Solution of the dual problem.
        """
        N, D = X.shape

        yy = y[:, None] @ y[:, None].T

        # Gram Matrix
        if self.kernel == 'quantum':
            if self.verbose: print('Computing Quantum Kernel ...')
            K = self.qk(X)
            if self.verbose:  print('Quantum Kernel computed!')
        else:
            K = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    K[i,j] = self.kernel_func(X[i], X[j])

        self.gramMatrix = K
        
        # QP solver 
        P = matrix(yy * K)
        q = matrix(-np.ones((N, 1)))

        if self.C is None: # hard SVM
            G = matrix((-np.eye(N)))
            h = matrix(np.zeros_like(y))
        else:  # soft SVM
            G = matrix(np.vstack((-np.eye(N), np.eye(N))))
            h = matrix(np.hstack((np.zeros_like(y), self.C*np.ones(N))))

        A = matrix(y.reshape(1,-1))
        b = matrix(np.zeros(1))
        #print('P:', P.size)
        #print('q:', q.size)
        #print('G:', G.size)
        #print('h:', h.size)
        #print('A:', A.size)
        #print('b:', b.size)
        # QP solver 
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        # lagrangian multipliers 
        alphas = np.ravel(solution['x'])
        
        # find the instances where the langrangian multipliers are non-zero
        is_sv = alphas > self.alpha_tol
        sv_ind = np.arange(len(alphas))[is_sv]
        self.alphas = alphas[is_sv]
        self.sv_X = X[is_sv]
        self.sv_y = y[is_sv]

        # bias 
        self.b = 0
        for i in range(len(self.alphas)):
            self.b += self.sv_y[i] # sum of all alphas 
            self.b -= np.sum(self.alphas * self.sv_y * self.gramMatrix[sv_ind[i], is_sv]) # sum per row 
        self.b /= len(self.alphas) # divided by alphas 

        # Compute w only if the kernel is linear
        if self.kernel == linear_kernel:
            #self.w = np.zeros(N)
            #for i in range(len(self.alphas)):
            #    self.w += self.alphas[i] * self.sv_X[i] * self.sv_y[i]
            self.w = np.einsum('i,i,ij', self.alphas, self.sv_y, self.sv_X)
        else:
            self.w = None

        self.is_fit = True

        if self.verbose:
            print(f'Found {len(self.alphas)} Support Vectors out of {len(X)} data points')
            
            if self.w is not None:
                print(f'Weights: {self.w}')
                print(f'Bias: {self.b}')
        if self.verbose: print('\n')
                
        # accuarcy 
        y_pred = self.predict(X)
        self.predictions_train = y_pred
        time.sleep(0.2)
        accuracy(y, y_pred, self.verbose, mode='training')
    
    #@jit
    def project(self, X):
        # If the model is not fit, raise an exception
        if not self.is_fit:
            raise SVMNotFitError
        # If the kernel is linear and 'w' is defined, the value of f(x) is determined by
        #   f(x) = X * w + b
        if self.w is not None:
            self.projections_train = np.dot(X, self.w) + self.b
            return np.dot(X, self.w) + self.b
        #elif self.kernel == 'quantum':

        #    return print(self.alphas.shape, self.sv_X.shape, self.sv_y.shape, X.shape)
        else:
            # Otherwise, it is determined by
            #   f(x) = sum_i{sum_sv{lambda_sv y_sv K(x_i, x_sv)}}
            y_pred = np.zeros(len(X))
            if self.verbose:
                with tqdm(range(len(X)), unit="batch") as comput_range:
                    for k in comput_range:
                        #comput_range.set_description(f"Epoch [{epoch+1}/{self.epochs}]  Training")
                        #print(f'{k+1}/{len(X)}')
                        for a, sv_X, sv_y in zip(self.alphas, self.sv_X, self.sv_y):
                            if self.kernel == 'quantum':
                                #print('a', a)
                                #print('sv_y', sv_y)
                                #print('self.qk(X[k], sv_X)', self.qk(X[k], sv_X))
                                y_pred[k] += a * sv_y * self.qk(X[k], sv_X)
                            else:
                                y_pred[k] += a * sv_y * self.kernel_func(X[k], sv_X)
            else:
                for k in range(len(X)):
                    #comput_range.set_description(f"Epoch [{epoch+1}/{self.epochs}]  Training")
                    #print(f'{k+1}/{len(X)}')
                    for a, sv_X, sv_y in zip(self.alphas, self.sv_X, self.sv_y):
                        if self.kernel == 'quantum':
                            #print('a', a)
                            #print('sv_y', sv_y)
                            #print('self.qk(X[k], sv_X)', self.qk(X[k], sv_X))
                            y_pred[k] += a * sv_y * self.qk(X[k], sv_X)
                        else:
                            y_pred[k] += a * sv_y * self.kernel_func(X[k], sv_X)
            self.decision_function = y_pred + self.b
            return y_pred + self.b

    def predict(self, X):
        return np.sign(self.project(X))

    def parameters(self):
        """ Gets all the relevant parameter for the return dictionary """
        self.params['weights'] = self.w
        self.params['bias'] = self.b
        self.params['sv_X'] = self.sv_X
        self.params['sv_y'] = self.sv_y
        self.params['sv_alphas'] = self.sv_alphas
        self.params['slope'] = self.slope
        self.params['intercept'] = self.intercept
        
        return self.params
    
    def kernel_func(self, x1, x2, X=None, y=None):
        if self.kernel == 'linear':
            return linear_kernel(x1, x2)
        elif self.kernel == 'polynominal':
            return polynomial_kernel(x2, x1, self.degree)
        elif self.kernel == 'gaussian':
            return gaussian_kernel(x1, x2, self.gamma)
        elif self.kernel == 'rbf':
            return rbf_kernel(x1, x2, self.gamma)
        elif self.kernel == 'sigmoid':
            return sigmoid_kernel(x1, x2, self.gamma)
        else:
            raise KernelError

    def plot(self, X, y, cmap, sv=True):
        """ 
        Params:
        ------
        X : 
        y : 
        clf : classifier
        """
        fig, ax = plt.subplots(1, figsize=(7,5))
        im = ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, alpha=0.7, label='Data')
        x1_max = X[:,0].max()
        x2_max = X[:,1].max()
        print(x1_max, x2_max)
        if self.kernel == 'quantum':
            X1, X2 = np.meshgrid(np.linspace(0, x1_max, 20), np.linspace(0, x2_max, 20))
        else:
            X1, X2 = np.meshgrid(np.linspace(0, x1_max, 50), np.linspace(0, x2_max, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = self.project(X).reshape(X1.shape)
        ax.contourf(X1, X2, Z, [-10.0, 0.0, 10.0], cmap=cmap, alpha=0.1)
        ax.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower', linestyles='-')
        ax.contour(X1, X2, Z + 1, [0.0], colors='k', linewidths=1, origin='lower', linestyles='--')
        ax.contour(X1, X2, Z - 1, [0.0], colors='k', linewidths=1, origin='lower', linestyles='--')

        if sv:
            ax.scatter(
                self.sv_X[:,0],
                self.sv_X[:,1],
                s=100,
                linewidth=1,
                facecolors="none",
                edgecolors="k",
                label="Support Vectors"
            )

        ax.axis("tight")

        cb = plt.colorbar(im, ax=ax)
        loc = np.arange(-1,1,1)
        cb.set_ticks(loc)
        cb.set_ticklabels(['-1','1'])
        plt.show()
    
    def get_kernel_matrix(self):
        return self.gramMatrix
    
    def plot_confusion_matrix(self, y, y_pred_custom, classes, title=None):
        cm = confusion_matrix(y, y_pred_custom)
        ax = sns.heatmap(cm, xticklabels=classes, yticklabels=classes, annot=True, fmt='0.2g', cmap=plt.cm.Blues,)
        
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
    
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Test Set')
        if title:
            ax.set_title(title, fontsize=12)
        
        plt.show()

class SVMNotFitError(Exception): # change text
    """Exception raised when the 'project' or the 'predict' method of an SVM object is called without fitting
    the model beforehand."""
    #print("Your model has not been fitted yet!")

class KernelError(Exception): 
    """Kernel does not exits! Choose one of the following 
        - linear \
        - polynominal \
        - gaussiam \
        - rbf \
        - sigmoid \
        - quantum """
    #print("""Kernel does not exits! Choose one of the following 
    #    - linear 
    #    - polynominal 
    #    - gaussiam  
    #    - rbf
    #    - sigmoid
    #    - quantum """)