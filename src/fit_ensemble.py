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
import abc
import numpy as np
import scipy.stats, scipy.io,scipy.optimize, scipy.misc,scipy.linalg,scipy.sparse
import pymc

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
        
    
class LVBP():
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,f_sim,f_exp,bootstrap_index_list,alpha0=None,uniform_prior_pops=True):
        self.D = f_sim - f_exp
        self.prior_pops = None
        self.f_sim = f_sim
        self.f_exp = f_exp        
        self.bootstrap_index_list = bootstrap_index_list
        
        self.m,self.n = f_sim.shape
                  
        if alpha0 == None:
            alpha0 = np.zeros(self.n)
        
        if uniform_prior_pops == True:
            self.prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(self.bootstrap_index_list)),value=np.ones(len(bootstrap_index_list) - 1) / float(len(bootstrap_index_list)))
        else:
            self.prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(self.bootstrap_index_list)))

        self.prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(self.bootstrap_index_list)),value=self.prior_dirichlet.value,observed=True)
            
    def sample(self,num_samples,thin=1,burn=0):        
        self.S = pymc.MCMC(self.variables)
        self.S.sample(num_samples,thin=thin,burn=burn)
        
    def accumulate_populations(self):
        a0 = self.S.trace("alpha")[:]
        n,m = self.f_sim.shape
        p = np.zeros(n)
        chi2 = []
        for i,a in enumerate(a0):
            pi = populations(a,self.f_sim,self.f_exp,self.prior_pops.value)
            p += pi
            means = pi.dot(self.D)
            chi2.append(np.linalg.norm(means)**2.)

            
        p /= p.sum()
        chi2 = np.mean(chi2)

        return p,chi2


class Gaussian_LVBP(LVBP):

    def __init__(self,f_sim,f_exp,regularization_strength,bootstrap_index_list,precision=None,save_pops=False,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,f_sim,f_exp,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(f_sim.T)

        alpha = pymc.MvNormal("alpha",np.zeros(self.n),tau=precision*regularization_strength,value=alpha0)

        @pymc.dtrm
        def prior_pops(prior_dirichlet=self.prior_dirichlet):
            return dirichlet_to_prior_pops(prior_dirichlet,self.bootstrap_index_list,len(self.f_sim))
        
        @pymc.dtrm
        def pi(alpha=alpha,prior_pops=prior_pops):
            return populations(alpha,f_sim,f_exp,prior_pops)
        
        @pymc.dtrm
        def mu(pi=pi):
            return pi.dot(self.D)
        
        @pymc.potential
        def logp(pi=pi,mu=mu):
            return -1*np.linalg.norm(mu)**2.

        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
            
        self.variables = [logp,alpha,pi,mu,prior_pops,self.prior_dirichlet]
        self.pi, self.prior_pops, self.alpha = pi , prior_pops, alpha
        
class Jeffreys_LVBP(LVBP):
    def __init__(self,f_sim,f_exp,bootstrap_index_list,save_pops=False,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,f_sim,f_exp,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        alpha = pymc.Uninformative("alpha",value=np.zeros(self.n))
        
        @pymc.dtrm
        def prior_pops(prior_dirichlet=self.prior_dirichlet):
            return dirichlet_to_prior_pops(prior_dirichlet,self.bootstrap_index_list,len(self.f_sim))
        
        @pymc.dtrm
        def pi(alpha=alpha,prior_pops=prior_pops):
            return populations(alpha,f_sim,f_exp,prior_pops)
        
        @pymc.dtrm
        def mu(pi=pi):
            return pi.dot(self.D)
            
        @pymc.potential
        def logp(pi=pi,mu=mu):
            return -1*np.linalg.norm(mu)**2.

        @pymc.potential
        def jeff(pi=pi,mu=mu):
            return log_jeffreys(pi,f_sim,mu=mu)
        
        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
                    
        self.variables = [logp,alpha,pi,mu,prior_pops,self.prior_dirichlet,jeff]
        self.pi, self.prior_pops, self.alpha = pi , prior_pops, alpha
        

def dirichlet_to_prior_pops(dirichlet,bootstrap_index_list,m):
    x = np.ones(m)

    pops = np.zeros(len(bootstrap_index_list))
    pops[:-1] = dirichlet[:]
    pops[-1] = 1.0 - pops.sum()

    for k,ind in enumerate(bootstrap_index_list):
        x[ind] = pops[k] / len(ind)
    
    return x
    
def cross_validated_mcmc(f_sim,f_exp,regularization_strength,bootstrap_index_list,num_samples = 50000,thin=1):
    all_indices = np.concatenate(bootstrap_index_list)
    test_chi = []
    train_chi = []
    precision = np.cov(f_sim.T)
    for j, train_ind in enumerate(bootstrap_index_list):
        test_ind = np.setdiff1d(all_indices,train_ind)
        num_local_block = 2
        local_bootstrap_index_list = np.array_split(np.arange(len(train_ind)),num_local_block)
        S = Gaussian_LVBP(f_sim[train_ind],f_exp,regularization_strength,local_bootstrap_index_list,precision=precision,uniform_prior_pops=True)
        test_chi_observable = pymc.Lambda("test_chi_observable",lambda alpha=S.alpha: chi2(alpha,f_sim[test_ind],f_exp))
        train_chi_observable = pymc.Lambda("train_chi_observable",lambda alpha=S.alpha: chi2(alpha,f_sim[train_ind],f_exp))
        S.variables.append(test_chi_observable)
        S.variables.append(train_chi_observable)
        S.sample(num_samples,thin=thin)
        test_chi.append(S.S.trace("test_chi_observable")[:].mean())
        train_chi.append(S.S.trace("train_chi_observable")[:].mean())

    test_chi = np.array(test_chi)
    train_chi = np.array(train_chi)
    print regularization_strength, train_chi.mean(), test_chi.mean()
    return test_chi,train_chi
            
def fisher_information(populations,f_sim,mu=None):
    if mu == None:
        mu = f_sim.T.dot(populations)
    D = f_sim - mu
    n = len(populations)
    Dia = scipy.sparse.dia_matrix((populations,0),(n,n))
    D = Dia.dot(D)
    S = D.T.dot(f_sim)
    return S

def log_jeffreys(populations,f_sim,mu=None):
    I = fisher_information(populations,f_sim,mu=mu)
    sign,logdet = np.linalg.slogdet(I)
    return 0.5 * logdet