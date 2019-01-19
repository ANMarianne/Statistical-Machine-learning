import numpy as np
from scipy import stats

class GaussianMixtureModel():
    """Density estimation with Gaussian Mixture Models (GMM).

    You can add new functions if you find it useful, but **do not** change
    the names or argument lists of the functions provided.
    """
    def __init__(self, X, K):
        """Initialise GMM class.

        Arguments:
          X -- data, N x D array
          K -- number of mixture components, int
        """
        self.X = X
        self.n = X.shape[0]
        self.D = X.shape[1]
        self.K = K


    def E_step(self, mu, S, pi):
        """Compute the E step of the EM algorithm.

        Arguments:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array

        Returns:
          r_new -- updated component responsabilities, N x K array
        """
        # Assert that all arguments have the right shape
        assert(mu.shape == (self.K, self.D) and\
               S.shape  == (self.K, self.D, self.D) and\
               pi.shape == (self.K, 1))
        r_new = np.zeros((self.n, self.K))

        # Task 1: implement the E step and return updated responsabilities
        # Write your code from here...
        for i in range(self.n):
            som=0
            for k in range(self.K):
                som+=pi[k]*stats.multivariate_normal.pdf(self.X[i,:],mu[k,:],S[k,:,:])
            for k in range(self.K):
                r_new[i,k]=pi[k]*stats.multivariate_normal.pdf(self.X[i,:],mu[k,:],S[k,:,:])/som 

        # ... to here.
        assert(r_new.shape == (self.n, self.K))
        return r_new


    def M_step(self, mu, r):
        """Compute the M step of the EM algorithm.

        Arguments:
          mu -- previous component means, K x D array
          r -- previous component responsabilities,  N x K array

        Returns:
          mu_new -- updated component means, K x D array
          S_new -- updated component covariances, K x D x D array
          pi_new -- updated component weights, K x 1 array
        """
        assert(mu.shape == (self.K, self.D) and\
               r.shape  == (self.n, self.K))
        mu_new = np.zeros((self.K, self.D))
        S_new  = np.zeros((self.K, self.D, self.D))
        pi_new = np.zeros((self.K, 1))

        # Task 2: implement the M step and return updated mixture parameters
        # Write your code from here...
        for k in range(self.K):
            Nk=0
            som1=np.zeros((self.K, self.D))
            som2=np.zeros((self.K, self.D, self.D))
            for i in range(self.n):
                Nk+=r[i,k]
                som1[k,:]+=r[i,k]*self.X[i,:]
            mu_new[k,:]=som1[k,:]/Nk
            for i in range(self.n):
                C = np.reshape(self.X[i,:]-mu_new[k,:], (-1,1))
                som2[k]+=r[i,k]*(C @ C.T)
            S_new[k,:]=som2[k,:,:]/Nk
            pi_new[k]=Nk/self.n

        # ... to here.
        assert(mu_new.shape == (self.K, self.D) and\
               S_new.shape  == (self.K, self.D, self.D) and\
               pi_new.shape == (self.K, 1))
        return mu_new, S_new, pi_new
    
    def negloglikelihood(self, mu, S, pi):
        """Compute the E step of the EM algorithm.

        Arguments:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array

        Returns:
          nlogl -- negative log-likelihood, 1x 1 array
        """
        # Assert that all arguments have the right shape
        assert(mu.shape == (self.K, self.D) and\
               S.shape  == (self.K, self.D, self.D) and\
               pi.shape == (self.K, 1))
        nlogl= 0
        l=0
        for i in range(self.n):
            for k in range(self.K):
                l+=pi[k]*stats.multivariate_normal.pdf(self.X[i,:],mu[k,:],S[k,:,:])
            nlogl+=(np.log(l))[0]
        return -nlogl

    def train(self, initial_params):
        """Fit a Gaussian Mixture Model (GMM) to the data in matrix X.

        Arguments:
          initial_params -- dictionary with fields 'mu', 'S', 'pi' and 'K'

        Returns:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array
          r -- component responsabilities, N x K array
        """
        # Assert that initial_params has all the necessary fields
        assert(all([k in initial_params for k in ['mu', 'S', 'pi']]))

        mu = np.zeros((self.K, self.D))
        S  = np.zeros((self.K, self.D, self.D))
        pi = np.zeros((self.K, 1))
        r  = np.zeros((self.n, self.K))

        # Task 3: implement the EM loop to train the GMM
        # Write your code from here...
        l0 =0
        l1 =0
        mu = initial_params["mu"]
        S = initial_params["S"]
        pi = initial_params["pi"]
        
        
        maxstep=100
        precision=1e-6
        
        r  = self.E_step( mu, S, pi)
        mu1, S1, pi1 =self.M_step(mu, r)
        l0  = self.negloglikelihood( mu, S, pi)
        l1  =  self.negloglikelihood( mu1 , S1, pi1)
        actual_precision = np.abs(l0-l1)
        nstep = 0
        
        while(actual_precision > precision or nstep < maxstep):
            r  = self.E_step(mu, S, pi)
            mu1, S1, pi1 = self.M_step(mu, r)
            l0  =  self.negloglikelihood(mu, S, pi)
            l1  =  self.negloglikelihood(mu1 , S1, pi1)
            actual_precision = l0-l1
            mu = mu1
            S  = S1
            pi = pi1
            nstep +=1


        # ... to here.
        assert(mu.shape == (self.K, self.D) and\
               S.shape  == (self.K, self.D, self.D) and\
               pi.shape == (self.K, 1) and\
               r.shape  == (self.n, self.K))
        return mu, S, pi, r


if __name__ == '__main__':
    np.random.seed(43)

    ##########################
    # You can put your tests here 
    #
    # 
    # 
    # 
    ##########################

