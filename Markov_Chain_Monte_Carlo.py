import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


# ##############################################################################
# Returns a single sample from the conditional distribution p(beta_0 | beta_1, beta_2, tau, mu_0, tau_0, X, y)
# ##############################################################################
def sample_beta_0(y, x, beta_1, beta_2, tau, mu_0, tau_0):
    N = len(y)
    assert len(x) == N
    # TODO: Task 1 - Implement a sample from the conditional distribution for beta_0
    precision = tau * N + tau_0 
    mean = (tau*sum(y) + tau_0*mu_0  - tau * sum((beta_1*x + beta_2*(x**2)))) / precision
    return np.random.normal(mean, 1.0 / np.sqrt(precision))

# ##############################################################################
# Returns a single sample from the conditional distribution p(beta_1 | beta_0, beta_2, tau, mu_1, tau_1, X, y)
# ##############################################################################
def sample_beta_1(y, x, beta_0, beta_2, tau, mu_1, tau_1):
    N = len(y)
    assert len(x) == N
    # TODO: Task 2 - Implement a sample from the conditional distribution for beta_1
    precision = tau * sum(x**2) + tau_1 
    mean = (tau*sum(x*y) + tau_1*mu_1  - tau * sum(x*(beta_0+beta_2*x**2))) / precision
    return np.random.normal(mean, 1 / np.sqrt(precision))

# ##############################################################################
# Returns a single sample from the conditional distribution p(beta_2 | beta_0, beta_1, tau, mu_2, tau_2, X, y)
# ##############################################################################
def sample_beta_2(y, x, beta_0, beta_1, tau, mu_2, tau_2):
    N = len(y)
    assert len(x) == N
    # TODO: Task 3 - Implement a sample from the conditional distribution for beta_2
    precision = tau * sum(x**4) + tau_2 
    mean = (tau*sum((x**2)*y) + tau_2*mu_2  - tau * sum((x**2)*(beta_0+beta_1*x))) / precision
    return np.random.normal(mean, 1 / np.sqrt(precision))

# ##############################################################################
# Returns a single sample from the conditional distribution p(tau | beta_0, beta_1, beta_2, alpha, beta, X, y)
# ##############################################################################
def sample_tau(y, x, beta_0, beta_1, beta_2, alpha, beta):
    N = len(y)
    # TODO: Task 4 - Implement a sample from the conditional distribution for tau
    alpha_new = (N/2) + alpha
    beta_new = (beta + (sum((y-(beta_0 + beta_1 * x + beta_2 * (x**2)))**2))/2)
    return np.random.gamma(alpha_new, 1 / beta_new)

# ##############################################################################
# Performs Gibbs sampling using the conditional distributions and returns the
# trace of the sampling process
# ##############################################################################
def gibbs(y, x, iters, init, hypers):
    assert len(y) == len(x)
    beta_0 = init["beta_0"]
    beta_1 = init["beta_1"]
    beta_2 = init["beta_2"]
    tau = init["tau"]
    mu_0   = hypers["mu_0"]
    tau_0  = hypers["tau_0"]
    mu_1   = hypers["mu_1"]
    tau_1  = hypers["tau_1"]
    mu_2   = hypers["mu_2"]
    tau_2  = hypers["tau_2"]
    alpha  = hypers["alpha"]
    beta   = hypers["beta"]

    trace = np.zeros((iters, 4))  ## trace to store values of beta_0, beta_1, beta_2, tau

    for it in range(iters):
        # TODO: Task 5 - Implement a Gibbs sampler
        beta_0 = sample_beta_0(y, x, beta_1, beta_2, tau, mu_0, tau_0)
        beta_1 = sample_beta_1(y, x, beta_0, beta_2, tau, mu_1, tau_1)
        beta_2 = sample_beta_2(y, x, beta_0, beta_1, tau, mu_2, tau_2)
        tau    = sample_tau(y, x, beta_0, beta_1, beta_2, alpha, beta)
        trace[it, :] = np.array((beta_0, beta_1, beta_2, tau))

    trace = pd.DataFrame(trace)
    trace.columns = ['beta_0', 'beta_1', 'beta_2', 'tau']

    return trace

if __name__ == '__main__':
    print('Sampling Coursework')
    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################

    beta_0_true = -1
    beta_1_true = 2
    beta_2_true = 0.5
    tau_true = 0.5

    N = 2000
    x = np.random.uniform(low=-3, high=6, size=N)
    y = np.random.normal(beta_0_true + beta_1_true * x + beta_2_true * x ** 2, 1 / np.sqrt(tau_true))

    ## specify initial values
    init = {"beta_0": 0,
            "beta_1": 0,
            "beta_2": 0,
            "tau": 2}

    ## specify hyper parameters
    hypers = {"mu_0": 0,
              "tau_0": 1,
              "mu_1": 0,
              "tau_1": 1,
              "mu_2": 0,
              "tau_2": 1,
              "alpha": 2,
              "beta": 1}

    iters = 1000

    trace = gibbs(y, x, iters, init, hypers)

    trace_burnt = trace[int(len(trace) / 2):]


    factor = 3.

    beta_0_med = trace_burnt['beta_0'].median()
    beta_0_std = trace_burnt['beta_0'].std()
    beta_0_low = beta_0_med - factor * beta_0_std
    beta_0_hi = beta_0_med + factor * beta_0_std
    print('Beta 0 fit: %s\n%f < %f < %f' % ('correct' if (beta_0_low
                                            < beta_0_true
                                            < beta_0_hi) else 'incorrect', beta_0_low, beta_0_true, beta_0_hi))

    beta_1_med = trace_burnt['beta_1'].median()
    beta_1_std = trace_burnt['beta_1'].std()
    beta_1_low = beta_1_med - factor * beta_1_std
    beta_1_hi = beta_1_med + factor * beta_1_std
    print('Beta 1 fit: %s\n%f < %f < %f' % ('correct' if (beta_1_low
                                                          < beta_1_true
                                                          < beta_1_hi) else 'incorrect', beta_1_low, beta_1_true,
                                            beta_1_hi))
    beta_2_med = trace_burnt['beta_2'].median()
    beta_2_std = trace_burnt['beta_2'].std()
    beta_2_low = beta_2_med - factor * beta_2_std
    beta_2_hi = beta_2_med + factor * beta_2_std
    print('Beta 2 fit: %s\n%f < %f < %f' % ('correct' if (beta_2_low
                                                          < beta_2_true
                                                          < beta_2_hi) else 'incorrect', beta_2_low, beta_2_true,
                                            beta_2_hi))
    tau_med = trace_burnt['tau'].median()
    tau_std = trace_burnt['tau'].std()
    tau_low = tau_med - factor * tau_std
    tau_hi = tau_med + factor * tau_std
    print('Tau fit: %s\n%f < %f < %f' % ('correct' if (tau_low
                                                          < tau_true
                                                          < tau_hi) else 'incorrect', tau_low, tau_true,
                                            tau_hi))
