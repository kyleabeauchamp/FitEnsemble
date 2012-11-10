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

import fit_ensemble

def identify_outliers(f_sim,f_exp):
    outliers = []
    for i,x in enumerate(f_sim.T):
        mu = f_exp[i]
        m = x.min()
        M = x.max()
        if mu >=m and mu <=M:
            pass
        else:
            outliers.append(i)
    outliers = np.array(outliers)
    return outliers

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
    pi = fit_ensemble.populations(alpha,f_sim,f_exp)
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



def relent(alpha,f_sim,f_exp):
    prior_pops = np.ones(len(f_sim))
    prior_pops /= prior_pops.sum()
    
    pi = populations(alpha,f_sim,f_exp,prior_pops)
    return np.log(pi).dot(pi)
    
def drelent(alpha,f_sim,f_exp):
    prior_pops = np.ones(len(f_sim))
    prior_pops /= prior_pops.sum()
    
    pi = populations(alpha,f_sim,f_exp,prior_pops)
    q = f_sim.T.dot(pi)
    grad = q * np.log(pi).dot(pi)
    
    L = pi*np.log(pi)
    grad -= f_sim.T.dot(L)

    return grad
    