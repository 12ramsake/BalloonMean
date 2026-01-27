"""
     Robust Estimation of Mean Vector
"""
# This is taken from the code repository - I did not write this code, see Yu et al
import numpy as np
from numpy import linalg as LA
import numpy.random as rgt
import math
from scipy.stats import norm



class m_est():
    methods = ["Catoni", "Huber"]

    def __init__(self, X):
        '''
        Argumemnts
            X: n by d numpy array. n is the number of observations.
               Each row is an observation vector of dimension d.
        '''
        self.n, self.d = X.shape
        self.X = X
        self.tau = (self.n/np.log(self.n))**0.5

    def priv_hist(self, epsilon = 0.5, delta = None, K = None):
        '''
        Return noisy estimate of $\EE\|X - \mu\|_2^2$.
        '''
        '''
        (epsilon, delta) : privacy parameters; default epsilon is 0.5 and 
                           delta is calculated based on the relationship between (epsilon, delta)-DP and epsilon-GDP
        K : number of bins; default is sqrt(n)/2
        '''

        if K == None:
            K = math.floor(np.sqrt(self.n)/2)

        if delta == None :
            delta = norm.cdf(-1 + epsilon/2) - np.exp(epsilon)*norm.cdf(-1 - epsilon/2)

        distances = [(LA.norm(self.X[2*i,:] - self.X[2*i + 1])**2)/2 for i in range(math.floor(self.n/2) - 1)]
        partition_size = len(distances) // K
        partitions = [distances[i:i+partition_size] for i in range(0, len(distances), partition_size)]
        medians = [np.median(partition) for partition in partitions]

        smallest_median = min(medians)
        largest_median = max(medians)

        smallest_bin = math.floor(np.log2(smallest_median))
        largest_bin = math.floor(np.log2(largest_median))

        count = np.zeros(largest_bin - smallest_bin + 1)

        for median in medians :
            bin_exponent = math.floor(np.log2(median))
            count[bin_exponent - smallest_bin] += 1

        prob = count/len(medians)
        t = 2*np.log(2/delta)/(epsilon*K) + 1/K

        for i in range(len(prob)):
            if prob[i] > 0:
                prob[i] += rgt.laplace(0, 2/(epsilon*K))
                if prob[i] < t:
                    prob[i] = 0

        return 2.0**(smallest_bin + np.argmax(prob))


    def robust_weight(self, x, method="Huber"):
        '''
        Compute the robust weight function
        '''
        if method == "Huber":
            return np.where(x>1, 1/x, 1)
        if method == "Catoni":
            return np.nan_to_num(np.log(1 + x + 0.5*x**2)/x, nan=1)

    def m_q(self, res, q = 4):
        '''
        Compute the robust q-norm of the residual matrix
        '''
        mq = np.median(np.linalg.norm(res, axis = 1)**q)
        return mq

    def Minsker_mean(self, mu0=np.array([]), max_iter=500, tol=1e-10):
        '''
        Compute the Minsker mean
        '''
        '''
        mu0: initial value of the mean vector
        '''
        if len(mu0.tolist()) == 0:
            mu0 = np.mean(self.X, axis = 0)

        alpha, p = 7/18, 0.1
        delta = 0.05
        psi = (1-alpha)*np.log((1-alpha)/(1-p)) + alpha*np.log(alpha/p)
        k = int(np.log(1/delta)/psi) + 1
        Z = np.zeros((k, self.d))
        m = int(self.n/k)
        for i in range(k):
            if i == k-1:
                Z[i, :] = self.X[m*i:, :].mean(0)
            else :
                Z[i, :] = self.X[m*i:m*(i+1), :].mean(0)

        mu1 = mu0
        r0, count = 1,0
        while r0 > tol and count <= max_iter:  #use Weiszfeld's algorithm
            weight = np.zeros(k)
            for i in range(k):
                weight[i] = (Z[i, :] - mu1).dot(Z[i, :] - mu1)**(-0.5)

            weight = weight/np.sum(weight, axis = 0)
            mu2 = np.matmul(Z.T, weight)
            r0 = (mu1 - mu2).dot(mu1 - mu2)**0.5
            mu1 = mu2
            count += 1

        return mu1, count

    def adaptive_huber(self, mu0= np.array([]), q = 4, max_iter=500, tol=1e-10, gamma = None):
        '''
        Compute the adaptive Huber mean estimator
        '''
        '''
        mu0: initial value of the mean vector
        q: parameter related to the existence of the q-th moment
        max_iter: maximum number of iterations
        tol: tolerance level
        gamma: exponential parameter in the tau
        '''
        if len(mu0.tolist()) == 0 :
            mu0 = np.mean(self.X, axis=0)

        mu_seq = np.zeros([self.d, int(max_iter)+1])
        mu_seq[:,0] = mu0

        if gamma == None:
            gamma = 1/2
        tau0 = (self.n/np.log(self.n))**(gamma)
        res = mu0 - self.X
        tau = 0.2*tau0 * (self.m_q(res, q)**(1/q))
        weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
        mu1 = mu0
        r0, count = 1, 0

        while r0 > tol and count <= max_iter:
            grad1 = res.T.dot(weight)/self.n
            diff_mu = -grad1
            r0 = diff_mu.dot(diff_mu)
            mu1 += diff_mu
            mu_seq[:, count+1] = mu1
            res = mu1 - self.X
            tau = 0.2*tau0 * (self.m_q(res, q)**(1/q))
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
            count += 1

        return mu1, [res, weight, tau, count], mu_seq[:, :count + 1]

    def adaptive_hubercov(self, mu0=np.array([])):
        '''
        Compute the robust covariance matrix
        '''
        if len(mu0.tolist()) == 0:
            mu0, _, _ = self.adaptive_huber2()

        res = mu0 - self.X
        sigma = self.m_q(res, 4)**(1/2)
        xi = sigma*np.sqrt(self.n/np.log(self.d*self.n))
        weight = self.robust_weight(np.sum(res**2, axis=1)/xi, method="Huber")
        Sigma = (res*weight[:, None]).T.dot(res)/self.n

        return Sigma, xi

    def noisy_adaptive_huber(self, mu0=np.array([]), epsilon = 0.5, T = None, eta = 1, m = None):
        '''
        Compute the GDP Huber mean estimator
        '''
        '''
        mu0: initial value of the mean vector
        epsilon: privacy parameter
        T: number of iterations
        eta: step size
        m: initial value of the moment value; if None, it is calculated by the private histogram
        '''
        if len(mu0.tolist()) == 0:
            mu0 = np.zeros(self.d)

        if T == None:
            T = int((eta**(-2))*(np.log(self.n)))

        mu_seq = np.zeros([self.d, T+1])
        mu_seq[:,0] = mu0

        if m == None:
            m = self.priv_hist(epsilon = epsilon/((T+1)**0.5))
            mu1 = mu0
            res = mu1 - self.X
            tau = 0.5*(m**1/2)*(epsilon*self.n/(np.sqrt(self.n*(self.d + np.log(self.n)))))**(1/2)
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")

            count = 0

            while count < T:
                grad1 = res.T.dot(weight)/self.n
                noise = rgt.multivariate_normal(np.zeros(self.d), np.identity(self.d))
                diff_mu = -eta*grad1 + 2*(eta)*((T + 1)**0.5)*tau/(epsilon*self.n)*noise
                mu1 += diff_mu
                mu_seq[:, count+1] = mu1
                res = mu1 - self.X
                weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
                count += 1

        else :
            mu1 = mu0
            res = mu1 - self.X
            tau = 0.5*(m**1/2)*(epsilon*self.n/(np.sqrt(self.n*(self.d + np.log(self.n)))))**(1/2)
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")

            count = 0

            while count < T:
                grad1 = res.T.dot(weight)/self.n
                noise = rgt.multivariate_normal(np.zeros(self.d), np.identity(self.d))
                diff_mu = -eta*grad1 + 2*(eta)*((T)**0.5)*tau/(epsilon*self.n)*noise
                mu1 += diff_mu
                mu_seq[:, count+1] = mu1
                res = mu1 - self.X
                weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
                count += 1

        return mu1, [res, weight, tau, count], mu_seq
    

    def noisy_hubercov(self, epsilon = 0.5, mu0 = np.array([]), threshold = 0.2) :
        '''
        Compute the GDP robust covariance matrix
        '''
        if len(mu0.tolist()) == 0 :
            mu0, _, _ = self.noisy_adaptive_huber(epsilon = epsilon) 

        res = mu0 - self.X
        sigma = self.priv_hist(epsilon = epsilon/(2))
        xi = 10*(sigma)*np.sqrt(self.n/np.log(self.d*self.n))
        weight = self.robust_weight(np.sum(res**2, axis=1)/xi, method="Huber")
        Sigma = (res*weight[:, None]).T.dot(res)/self.n
        M = rgt.normal(size = (self.d,self.d))
        E = np.triu(M, k = 0) + np.triu(M, k = 1).T
        Sigma += 2*xi/(epsilon*self.n)*E

        eigenvalues, eigenvectors = LA.eigh(Sigma)
        truncated_eigenvalues = np.diag(np.maximum(eigenvalues, threshold))

        Sigma = np.dot(np.dot(eigenvectors, truncated_eigenvalues), eigenvectors.T)

        return Sigma , xi

    def noisy_trun_mean(self, mu0=np.array([]), epsilon = 0.5,R = None, delta = None):
        '''
        Implementation of the algorithm in Cai et al(2020)
        '''
        if R == None:
            R = np.amax(np.absolute(self.X))

        if delta == None :
            delta = 1/2

        Y = np.where(self.X > R, R, self.X)
        Y = np.where(Y < -R, -R, Y)

        if len(mu0.tolist()) == 0:
            mu0 = np.mean(Y, axis = 0)

        noise = rgt.multivariate_normal(np.zeros(self.d), np.identity(self.d))
        mu = mu0 + noise*2*R*((self.d*np.log(1/delta))**0.5)/(self.n*epsilon)
        return mu