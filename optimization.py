import numpy as np

class SGDo():
    """
    Implements algorithm 2 in "SGD without Rplacement: Sharper Rates for General Smooth Convex Functions"
    """
    def __init__(self, D, G, L, K, n, Mu, l, sc = False, Large_K = False)
        """
        Args:
          D: Diameter of the constraint set. We assume for simplicity that our constraint set is ||x||_2 <= D/2
          G: Lipschitz constant 
          L: Smoothness parameter ||grad_f(x) - grad_f(y)|| <= L||x-y||
          K: Number of passes over the entire dataset we want to optimize for
          n: Number of datapoints in the data set
          l: some extra parameter for the strongly convex cases (used in Theorem 1 and 2)
          Mu: Strong convexity parameter
          sc: True if we want to use the Strongly Convex version of the algorithm (Theorem 1 and 2), False for the general smooth convex function (Theorem 3)
          Large_K = when sc is True, this decides if we follow the step size in Theorem 1 (Large_K=True) or Theorem 2 (Large_K=False)
        """
        self.D = D
        self.G = G
        self.L = L 
        self.K = K 
        self.n = n 
        self.l = l
        self.Mu = Mu 
        self.sc = sc 
        self.Large_K = Large_K 
    
    def learning_rate(self):
        if self.sc:
            if self.Large_K:
                return 4*self.l*np.log(self.n*self.K)/self.Mu/self.n/self.K 
            else:
                return min(2/self.L, 4*self.l*np.log(self.n*self.K)/self.Mu/self.n/self.K )
        else:
            return min(2/self.L, self.D/self.G/np.sqrt(self.K*self.n))