import numpy as np

class SGDo:
    """
    Implements algorithm 2 in "SGD without Rplacement: Sharper Rates for General Smooth Convex Functions"
    We assume that the constraint set is ||x||_2 <= D/2 for easy projection step
    """
    def __init__(self, D, G, L, K, n, Mu, l, sc = False, Large_K = False):
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

    def run(self, f, dataset, init_x):
        """
        Runs the SGDo algorithm
        Returns a tuple of (last iterate, overall average, tail average, loss)
        """
        loss = []
        # initialize x
        x = init_x 
        alpha = self.learning_rate()

        average = 0
        tail_average = 0
        for k in range(self.K):
            # update the tail_average
            if k > self.K/2:
                tail_average += x

            
            perm = np.random.permutation(self.n)
            for idx in perm:
                # Take a gradient step
                grad_fi = f.grad_f(x, dataset[idx])
                x -= alpha*grad_fi
                # Projection
                if np.linalg.norm(x) > self.D/2:
                    x = x/np.linalg.norm(x)*self.D/2
                # update the average
                average += x
                
                # Record the overall loss
                loss.append(np.mean([f.f(x, d) for d in dataset]))

        return x, average/self.n/self.K, tail_average/(self.K - np.ceil(self.K/2) +1), loss
            


class LISGD:
    """Last Iterate SGD"""
    def __init__(self, G, D, Lambda, C, T, n, sc=False):
        """
        Args:
          G: Lipschitz
          D: Diameter
          Lambda: Strong convexity
          C: some arbitrary positive parameter in the algorithm
          T: number of iterations
          n = size of dataset
          sc = whether or not the loss function is strongly convex
        """
        self.G = G 
        self.D = D 
        self.Lambda = Lambda 
        self.C = C 
        self.T = T 
        self.n = n
        self.sc = sc

    def learning_rate(self):
        alpha = []

        T_im1 = 0 # T_{i-1}
        k = int(np.ceil(np.log2(self.T)))
        for i in range(1, k+2):
            if i==k+1:
                T_i = self.T
            else:
                T_i = self.T - np.ceil(self.T*2**(-i))

            if self.sc:
                alpha_i = list(2**(-i+1)/self.Lambda/np.arange(T_im1+1, T_i+1))
            else:
                alpha_i = [self.C * 2**(-i+1)/np.sqrt(T)]* (T_i - T_im1)
            alpha += alpha_i
            T_im1 = T_i
        assert len(alpha) == self.T
        return alpha 
    
    def run(self, f, dataset, init_x):
        """
        Runs the LISGD algorithm
        Returns a tuple of (last iterate, overall average, loss)
        """
        loss = []
        # initialize x
        x = init_x 
        alpha = self.learning_rate()

        average = 0
        for t in range(self.T):
            idx = np.random.choice(self.n)
            # Take a gradient step
            grad_fi = f.grad_f(x, dataset[idx])
            x -= alpha[t]*grad_fi
            # Projection
            if np.linalg.norm(x) > self.D/2:
                x = x/np.linalg.norm(x)*self.D/2
            # update the average
            average += x

            # Record the overall loss
            loss.append(np.mean([f.f(x, d) for d in dataset]))
        return x, average/self.T, loss
            


class SGD:
    """Vanilla SGD, for lipschitz, beta smooth, alpha sc and beta smooth, and strongly convex and lipschitz"""
    def __init__(self, L, alpha, beta, D, T, n, sigma, sc=False, Lipschitz = True, smooth=False):
        """
        Args:
          L: Lipschitz coefficient
          alpha: strong convexity
          beta: smoothness 
          sigma: variance bound on the gradient
          T: number of iterations
          n = size of dataset
          sc = whether or not the loss function is strongly convex
        """
        self.L = L
        self.alpha = alpha
        self.beta = beta 
        self.D = D
        self.T = T 
        self.n = n
        self.sigma = sigma
        self.sc = sc
        self.Lipschitz = Lipschitz 
        self.smooth = smooth 

    def learning_rate(self):

        if self.Lipschitz:
            if not self.sc and not self.smooth:
                eta = [self.D / np.sqrt(self.sigma**2+self.L**2)/ np.sqrt(self.T)] * self.T
            elif self.smooth and not self.sc:
                c = np.sqrt(2)*self.sigma/self.D
                eta = [1/(self.beta + c*np.sqrt(self.T))]*self.T
            elif self.sc and not self.smooth:
                eta = 1/self.alpha/np.arange(1, self.T+1)
            elif self.sc and self.smooth:
                eta = [np.log(self.T)/self.alpha/self.T]*self.T
            else:
                raise ValueError("There is no combination for lipschitz, strongly convex and smooth")
        else:
            raise ValueError("We are not considering non-lipschitz cases")
        assert len(eta) == self.T
        return eta
    
    def run(self, f, dataset, init_x):
        """
        Runs the LISGD algorithm
        Returns a tuple of (last iterate, overall average, loss)
        """
        loss = []
        # initialize x
        x = init_x 
        alpha = self.learning_rate()

        average = 0
        for t in range(self.T):
            idx = np.random.choice(self.n)
            # Take a gradient step
            grad_fi = f.grad_f(x, dataset[idx])
            x -= alpha[t]*grad_fi
            # Projection
            if np.linalg.norm(x) > self.D/2:
                x = x/np.linalg.norm(x)*self.D/2
            # update the average
            average += x

            # Record the overall loss
            loss.append(np.mean([f.f(x, d) for d in dataset]))
        return x, average/self.T, loss