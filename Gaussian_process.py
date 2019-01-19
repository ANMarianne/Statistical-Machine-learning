import numpy as np
from scipy.optimize import minimize


# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    sample = np.zeros((mean.shape[0], )) # This is only a placeholder
    # Task 2:
    # TODO: Implement a draw from a multivariate Gaussian here
    sample=np.random.multivariate_normal(mean.ravel(), cov, 1)
    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug

        n = X.shape[0]
        covMat = np.zeros((n,n))

        # Task 1:
        # TODO: Implement the covariance matrix here

        for j1 in range(n):
            for j2 in range(n):
                covMat[j1,j2]= self.sigma2_f*np.exp(-((np.linalg.norm(X[j1,:]-X[j2,:]))**2)/(2*(self.length_scale)**2 ))


        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)
        self.L = np.linalg.cholesky(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        self.L = np.linalg.cholesky(self.K)
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction

        Cv = self.k.covMatrix(self.X, Xa)
        n0=Cv.shape[0]
        n1=  Xa.shape[0]
        
        K1 = np.linalg.solve((self.L).T, np.linalg.solve(self.L,self.y)) 
        self.sigma2_f, self.length_scale, self.sigma2_n=self.k.getParamsExp()
        mean_fa = Cv[n0-n1:,:n0-n1] @ K1
        cov_fa = (Cv[n0-n1:,n0-n1:]-self.sigma2_n*np.eye(n1))- Cv[n0-n1:,:n0-n1]@ np.linalg.inv(self.K)@ ((Cv[n0-n1:,:n0-n1]).T)
       

        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        mll = 0
        K1=np.linalg.solve((self.L).T, np.linalg.solve(self.L,self.y))
        n= self.n
        pi=np.pi
        logdet=2*sum(np.log(np.diag(self.L)))
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        mll=0.5*(self.y).T @ K1  +0.5*logdet+0.5*n*np.log(2*pi)
        # Return mll
        mll = mll[0][0]
        return mll

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters

        K3=np.zeros((self.n,self.n))
        K4=np.zeros((self.n,self.n))
        K5=np.zeros((self.n,self.n))
        K1=np.linalg.solve((self.L).T, np.linalg.solve(self.L,self.y))

        K2=K1@((K1).T)-np.linalg.inv(K)

        self.sigma2_f, self.length_scale, self.sigma2_n= self.k.getParamsExp()

        for j1 in range(self.n):
             for j2 in range(self.n):
                coeff = ((np.linalg.norm(self.X[j1]-self.X[j2]))**2)/(2*((self.length_scale)**2))
                K3[j1,j2]= 2*self.sigma2_f *np.exp(-coeff)
                K5[j1,j2]= self.sigma2_f *np.exp(-coeff)*(2* coeff)

        grad_ln_length_scale= -0.5*np.trace(K2@ K5) 
        grad_ln_sigma_f= -0.5*np.trace(K2@ K3)
        K4= 2*self.sigma2_n*np.eye(self.n)
        grad_ln_sigma_n= -0.5*np.trace(K2@ K4)


        # Combine gradients
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        n=ya.shape[0]
        for i in range(n):
            mse+=((ya[i]-fbar[i])**2)
        mse=(1/n)*mse
        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        n5=ya.shape[0]
        pi=np.pi

        sigm=np.diag(cov)
        for i in range(n5):
            msll+=(0.5*(np.log(2*pi*sigm[i]))+(((ya[i]-fbar[i])**2)/(2*sigm[i])))
        msll=msll/ n5
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x

if __name__ == '__main__':

    np.random.seed(42)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
X_train, y_train, X_test, y_test = loadData('yacht_hydrodynamics.data')

k = RadialBasisFunction(params)

params=np.array([0.5*np.log(1.0),np.log(0.1),0.5*np.log(0.5)])

k1=GaussianProcessRegression(X_train, y_train,k)

mean, cov = k1.predict(X_test)


k1.optimize(params)


