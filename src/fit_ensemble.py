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

def log_partition(alpha,f_sim,f_exp):
    """Return the log partition function of an alpha-reweighted ensemble."""
    q = -1*f_sim.dot(alpha).astype('float128')
    return scipy.misc.logsumexp(q)

def dlog_partition(alpha,f_sim,f_exp):
    """Return the gradient of the alpha-reweighted log partition function.

    Notes
    -----
    This simply gives the alpha-reweighted ensemble average prediction of
    the experimental observables.  
    
    """
    pi = populations(alpha,f_sim,f_exp)
    grad = f_sim.T.dot(pi)
    return grad

def partition(alpha,f_sim,f_exp):
    """Return the partition function of an alpha-reweighted ensemble."""
    return np.exp(log_partition(alpha,f_sim,f_exp))
    
def objective_chodera(alpha,f_sim,f_exp):
    """Return the maximum entropy objective function from Pitera et al.

    Notes
    -----
    This is equation 15 in [1].  If your experimental data is consistent 
    with your simulated ensemble and the hessian is positive definite, then
    minimizing this objective function will lead to the unique maximum 
    entropy reweighted ensemble. 

    References
    ----------
    .. [1] Pitera, J, Chodera, J. "On the use of Experimental Observables to 
        Bias Simulated Ensembles." JCTC 2012.
    """
    f = log_partition(alpha,f_sim,f_exp) + f_exp.dot(alpha)
    return f
    
def dobjective_chodera(alpha,f_sim,f_exp):
    """Return the gradient of the maximum entropy objective function.

    Notes
    -----
    This is equation 16 in [1].  If your experimental data is consistent 
    with your simulated ensemble and the hessian is positive definite, then
    this gradient will equal zero at the unique maximum entropy reweighted 
    ensemble. 

    References
    ----------
    .. [1] Pitera, J, Chodera, J. "On the use of Experimental Observables to 
        Bias Simulated Ensembles." JCTC 2012.
    """
    return dlog_partition(alpha,f_sim,f_exp) + f_exp

def populations(alpha,f_sim,f_exp,prior_pops):
    """Return the reweighted conformational populations."""
    q = -1*f_sim.dot(alpha)
    q -= q.mean()
    pi = np.exp(q)
    pi *= prior_pops
    pi /= pi.sum()
    return pi

def dpopulations(alpha,f_sim,f_exp):
    """Return the derivative of reweighted conformational populations.
    
    Notes
    -----
    This returns a two-dimensional Numpy array.  Suppose that p[j] are the
    populations of each conformation.
    
    dp[j,i] = the partial derivative of p[j] with respect to alpha[i].  
    
    """
    prior_probs = np.ones(len(f_sim))
    prior_probs /= prior_probs.sum()
    pi = populations(alpha,f_sim,f_exp,prior_probs)
    v = f_sim.T.dot(pi)
    n = pi.shape[0]
    pi_diag = scipy.sparse.dia_matrix(([pi],[0]),(n,n))
    grad = -1*pi_diag.dot(f_sim)
    grad += np.outer(pi,v)
    return grad
    
def chi2(alpha,f_sim,f_exp,prior_pops):
    """Return the chi squared objective function.

    Notes
    -----

    References
    ----------
    .. [1] Beauchamp, K. A. 
    """
    pi = populations(alpha,f_sim,f_exp,prior_pops)
    q = f_sim.T.dot(pi)
    delta = f_exp - q

    d = scipy.sparse.dia_matrix((pi,0),shape=(len(pi),len(pi)))
    sigma = 1. + d.dot(f_sim).var(0) / len(pi)
    
    f = np.linalg.norm(delta / sigma)**2.
    return f

def dchi2(alpha,f_sim,f_exp):
    """Return the gradient of chi squared objective function.

    Notes
    -----

    References
    ----------
    .. [1] Beauchamp, K. A. 
    """
    prior_probs = np.ones(len(f_sim))
    prior_probs /= prior_probs.sum()
    pi = populations(alpha,f_sim,f_exp,prior_probs)
    delta = 2*(f_sim.T.dot(pi) - f_exp)
    q = dpopulations(alpha,f_sim,f_exp)
    v = q.T.dot(f_sim)
    return v.dot(delta)

def ridge(alpha):
    """Return the ridge (L2) regularization penalty."""
    return 0.5*np.linalg.norm(alpha)**2.

def dridge(alpha):
    """Return the gradient of the ridge (L2) regularization penalty."""
    return alpha

def minimize_chi2(alpha,f_sim,f_exp,regularization_strength, regularization_method="ridge"):
    """Find the coupling parameters alpha to match experimental ensemble.
    """
    prior_probs = np.ones(f_sim.shape[0]) / float(f_sim.shape[0])
        
    if regularization_method == "maxent":
        f = lambda x: chi2(x,f_sim,f_exp,prior_probs) + relent(x,f_sim,f_exp) * regularization_strength
        f0 = lambda x: chi2(x,f_sim,f_exp,prior_probs)
        df = lambda x: dchi2(x,f_sim,f_exp) + drelent(x,f_sim,f_exp) * regularization_strength
    elif regularization_method == "ridge":
        f = lambda x: chi2(x,f_sim,f_exp,prior_probs) + ridge(x) * regularization_strength
        f0 = lambda x: chi2(x,f_sim,f_exp,prior_probs)
        df = lambda x: dchi2(x,f_sim,f_exp)  + dridge(x) * regularization_strength
        
    print(regularization_strength,f(alpha),f0(alpha))
   
    alpha = scipy.optimize.fmin_l_bfgs_b(f,alpha,df,disp=True,maxfun=10000,factr=10.)[0]
    print("Final Objective function (regularized) = %f.  Final chi^2 = %f"%(f(alpha),f0(alpha)))
    return alpha
    
    
def relent(alpha,f_sim,f_exp):
    prior_probs = np.ones(len(f_sim))
    prior_probs /= prior_probs.sum()
    
    pi = populations(alpha,f_sim,f_exp,prior_probs)
    return np.log(pi).dot(pi)
    
def drelent(alpha,f_sim,f_exp):
    prior_probs = np.ones(len(f_sim))
    prior_probs /= prior_probs.sum()
    
    pi = populations(alpha,f_sim,f_exp,prior_probs)
    q = f_sim.T.dot(pi)
    grad = q * np.log(pi).dot(pi)
    
    L = pi*np.log(pi)
    grad -= f_sim.T.dot(L)

    return grad
    