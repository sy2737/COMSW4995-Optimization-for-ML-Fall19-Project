import numpy as np
from scipy.special import expit
class Function:
    """
    Template class for different loss functions 
    """
    def __init__(self, Lambda):
        self.Lambda = Lambda # regularization parameter
    def f(self, x, d):
        """ Evaluates the function at point x for datapoint d """
        raise NotImplementedError
    def grad_f(self, x, d):
        """ Returns the gradient of functino at point x for datapoint d """
        raise NotImplementedError 
    

class LogisticRegression(Function):
    """ Assume that d comes in a tuple: 
      d = (d_x, d_y)
      where d_x is a vector and d_y \in {-1,1}
    """
    def __init__(self, Lambda):
        self.Lambda = Lambda
    def f(self, x, d):
        return np.log(1+np.exp(-np.dot(x, d[0])*d[1])) + self.Lambda/2*np.linalg.norm(x)**2
    def grad_f(self, x,d):
        #c = np.exp(-np.dot(x, d[0])*d[1])
        #return -c/(1+c)*d[1] * d[0] + self.Lambda*x
        c = -np.dot(x, d[0])*d[1]
        return -expit(c)*d[1] * d[0] + self.Lambda*x





