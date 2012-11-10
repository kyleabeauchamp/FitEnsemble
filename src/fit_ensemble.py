"""

Most functions in this module take three inputs: alpha, f_sim, and f_exp.
All these functions share a common model framework.  We have run a simulation
and calculated a set of observables at each conformation.  Thus, f_sim[j,i]
gives the ith predicted observable at frame j.  We assume that our 
experimental data consists of n equilibrium measurements--the ith experiment
is stored in f_exp[i].  Finally, alpha is a vector of coupling parameters 
that will be used to reweight the simulation value using a biasing potential 
that is linear in the predicted observables: 
U(x_i) = \sum_j alpha[i] f_sim[j,i]

Finally, for best results one must ensure that f_sim and f_exp have been
normalized divided by the estimated uncerainty associated with each 
experimental comparison.  This uncertainty could be experimental uncertainty
or the uncertainty inherent in predicting the observable from a simulated
conformation.


"""

import numpy as np
import scipy.stats, scipy.io,scipy.optimize, scipy.misc

def get_prior_pops(n,prior_pops):
    """Helper function returns uniform distribution if prior_pops is None."""
    if prior_pops != None:
        return prior_pops
    else:
        x = np.ones(n)
        x /= x.sum()
        return x

def populations(alpha,f_sim,f_exp,prior_pops=None):
    """Return the reweighted conformational populations."""

    prior_pops = get_prior_pops(len(f_sim),prior_pops)
    
    q = -1*f_sim.dot(alpha)
    q -= q.mean()
    pi = np.exp(q)
    pi *= prior_pops
    pi /= pi.sum()
    return pi

def dpopulations(alpha,f_sim,f_exp,prior_pops=None):
    """Return the derivative of reweighted conformational populations.
    
    Notes
    -----
    This returns a two-dimensional Numpy array.  Suppose that p[j] are the
    populations of each conformation.
    
    dp[j,i] = the partial derivative of p[j] with respect to alpha[i].  
    
    """
    prior_pops = get_prior_pops(len(f_sim),prior_pops)
    pi = populations(alpha,f_sim,f_exp,prior_pops)
    v = f_sim.T.dot(pi)
    n = pi.shape[0]
    pi_diag = scipy.sparse.dia_matrix(([pi],[0]),(n,n))
    grad = -1*pi_diag.dot(f_sim)
    grad += np.outer(pi,v)
    return grad
    
def chi2(alpha,f_sim,f_exp,prior_pops = None):
    """Return the chi squared objective function.

    Notes
    -----

    References
    ----------
    .. [1] Beauchamp, K. A. 
    """
    prior_pops = get_prior_pops(len(f_sim),prior_pops)
    pi = populations(alpha,f_sim,f_exp,prior_pops)
    q = f_sim.T.dot(pi)
    delta = f_exp - q

    f = np.linalg.norm(delta)**2.
    return f

def dchi2(alpha,f_sim,f_exp,prior_pops = None):
    """Return the gradient of chi squared objective function.

    Notes
    -----

    References
    ----------
    .. [1] Beauchamp, K. A. 
    """
    
    prior_pops = get_prior_pops(len(f_sim),prior_pops)

    pi = populations(alpha,f_sim,f_exp,prior_pops)
    delta = (f_sim.T.dot(pi) - f_exp)
    dpi = dpopulations(alpha,f_sim,f_exp,prior_pops)
    
    r = dpi.T.dot(f_sim)
    grad = 2*r.dot(delta)
        
    return grad
        
def ridge(alpha):
    """Return the ridge (L2) regularization penalty."""
    return 0.5*np.linalg.norm(alpha)**2.

def dridge(alpha):
    """Return the gradient of the ridge (L2) regularization penalty."""
    return alpha

def minimize_chi2(alpha,f_sim,f_exp,regularization_strength, regularization_method="ridge",prior_pops=None):
    """Find the coupling parameters alpha to match experimental ensemble.
    """
    prior_pops = get_prior_pops(len(f_sim),prior_pops)
        
    if regularization_method == "ridge":
        f = lambda x: chi2(x,f_sim,f_exp,prior_pops) + ridge(x) * regularization_strength
        f0 = lambda x: chi2(x,f_sim,f_exp,prior_pops)
        df = lambda x: dchi2(x,f_sim,f_exp)  + dridge(x) * regularization_strength
    else:
        raise(Exception("Incorrect regularization_method ( = %s"%regularization_method))
        
    print(regularization_strength,f(alpha),f0(alpha))
   
    alpha = scipy.optimize.fmin_l_bfgs_b(f,alpha,df,disp=True,maxfun=10000,factr=10.)[0]
    print("Final Objective function (regularized) = %f.  Final chi^2 = %f"%(f(alpha),f0(alpha)))
    return alpha
    
    
