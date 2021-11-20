import numpy as np 
from cvxopt import matrix, solvers

from quantum_svm.utils.utils import accuracy

class linearSVC:
    def __init__(self, C=None, alpha_tol=1e-4, verbose=True):
        """ Linear Support Vector classifier implementing SVM dual problem 
        for soft and hard margins.
            * if C is None -> hard margin
            * if C is not None -> soft margin 

        Parameters
        ----------
        C : int
            determines the margin strength

        alpha_tol : float,
                    minimum alpha value
        
        verbose : bool
                  determines if report is printed
        """
        self.alphas = None
        self.alpha_tol = alpha_tol
        self.C = C
        if C is not None: self.C = float(self.C)
        self.verbose = verbose
        self.params = {}

        self.sv_alphas = None
        self.sv_X = None
        self.sv_y = None
        self.w = None
        self.b = None
        
        # for graphing
        self.intercept = None
        self.slope = None

        self.is_fit = False
        
    def fit(self, X, y):
        """ Solves SVM dual problem with a QP solver.
        Performs the fir for the classification.

        Parameters
        ----------
        X : array, shape [N, D]
            Input features.
        y : array, shape [N]
            Binary class labels (in {-1, 1} format).
        """
        N, D = X.shape

        yy = y[:, None] @ y[:, None].T
        XX = X @ X.T

        # - - - QP solver - - - 
        P = matrix(yy * XX)
        q = matrix(-np.ones((N, 1)))

        if self.C is None: # hard margin SVM
            G = matrix((-np.eye(N)))
            h = matrix(np.zeros_like(y))
        else:  # soft margin SVM
            G = matrix(np.vstack((-np.eye(N), np.eye(N))))
            h = matrix(np.hstack((np.zeros_like(y), self.C*np.ones(N))))

        A = matrix(y.reshape(1,-1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        # - - - QP solver - - - 

        # lagrangian multipliers
        self.alphas = np.ravel(solution['x'])

        # find the instances where the langrangian multipliers are non-zero
        is_sv = (self.alphas > self.alpha_tol).flatten()

        self.sv_alphas = self.alphas[is_sv]
        self.sv_X = X[is_sv]
        self.sv_y = y[is_sv]

        # weights 
        self.w = np.einsum('i,i,ij', self.sv_alphas.flatten(), self.sv_y, self.sv_X)

        # bias
        biases = y[is_sv] - np.dot(X[is_sv, :], self.w)
        self.b = np.sum(self.sv_alphas*biases) / np.sum(self.sv_alphas)
        
        # graphing
        self.slope = - self.w[0] / self.w[1]
        self.intercept = -self.b / self.w[1]

        self.is_fit = True 

        if self.verbose:
            print(f'Found {len(self.sv_alphas)} Support Vectors out of {len(X)} data points')
            print('\n')
            print(f'Weights: {self.w}')
            print(f'Bias: {self.b}')
            print(f'Decision Hyperplane: {self.slope} * x + {self.intercept} ')

        print('\n')
                
        # accuarcy 
        y_pred = self.predict(X)
        accuracy(y, y_pred, self.verbose, mode='train')

    def project(self, X):
        assert self.w.shape[0] == X.shape[1]
        return X @ self.w + self.b
    
    def predict(self, X):
        """ predicts the classes based on the classifier """
        if not self.is_fit:
            raise SVMNotFitError
        return np.sign(self.project(X))

    def compute_params(self):
        """ Gets all the relevant parameter for the return dictionary """
        self.params['weights'] = self.w
        self.params['bias'] = self.b
        self.params['sv_X'] = self.sv_X
        self.params['sv_y'] = self.sv_y
        self.params['sv_alphas'] = self.sv_alphas
        self.params['slope'] = self.slope
        self.params['intercept'] = self.intercept
        
        return self.params
    
    
class SVMNotFitError(Exception): # change text 
    """Exception raised when the 'project' or the 'predict' method of an SVM object is called without fitting
    the model beforehand.""" 