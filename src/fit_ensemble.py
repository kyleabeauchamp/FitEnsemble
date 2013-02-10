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
    pi_sum = pi.sum()
    if np.isnan(pi_sum) or np.isinf(pi_sum):
        pi = np.zeros(len(pi))
        pi[q.argmax()] = 1.
    else:
        pi /= pi_sum
    return pi
    
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

        
    
class LVBP():
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,f_sim,f_exp,sigma_vector,bootstrap_index_list,alpha0=None,uniform_prior_pops=True):
        self.D = f_sim - f_exp
        self.sigma_vector = sigma_vector
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
            chi2.append(np.linalg.norm(means / self.sigma_vector)**2.)

            
        p /= p.sum()
        chi2 = np.mean(chi2)
        rms = chi2 / float(m)

        return p,rms

class Gaussian_LVBP(LVBP):

    def __init__(self,f_sim,f_exp,sigma_vector,regularization_strength,bootstrap_index_list,precision=None,save_pops=False,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,f_sim,f_exp,sigma_vector,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(f_sim.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))

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
            return -1*np.linalg.norm(mu / sigma_vector)**2.

        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
            
        self.variables = [logp,alpha,pi,mu,prior_pops,self.prior_dirichlet]
        self.pi, self.prior_pops, self.alpha = pi , prior_pops, alpha

class MaxEnt_LVBP(LVBP):

    def __init__(self,f_sim,f_exp,sigma_vector,regularization_strength,bootstrap_index_list,precision=None,save_pops=False,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,f_sim,f_exp,sigma_vector,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(f_sim.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))

        alpha = pymc.Uninformative("alpha",value=np.zeros(len(f_exp)))

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
            if pi.min() <= 0:
                return -1*np.inf
            else:
                return -1*np.linalg.norm(mu / sigma_vector)**2. - regularization_strength*sum((pi * np.log(pi)))

        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
            
        self.variables = [logp,alpha,pi,mu,prior_pops,self.prior_dirichlet]
        self.pi, self.prior_pops, self.alpha = pi , prior_pops, alpha


class Gaussian_LVBP_Bias(LVBP):

    def __init__(self,f_sim,f_exp,sigma_vector,regularization_strength,bootstrap_index_list,precision=None,save_pops=False,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,f_sim,f_exp,sigma_vector,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(f_sim.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))

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
            return -1*np.linalg.norm(mu / sigma_vector)**2.

        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
            
        self.variables = [logp,alpha,pi,mu,prior_pops,self.prior_dirichlet]
        self.pi, self.prior_pops, self.alpha = pi , prior_pops, alpha

class Gaussian_Corr_LVBP(LVBP):

    def __init__(self,f_sim,f_exp,sigma_vector,regularization_strength,bootstrap_index_list,precision=None,save_pops=False,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,f_sim,f_exp,sigma_vector,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(f_sim.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))

        rho = np.corrcoef(f_sim.T)
        rho_inverse = np.linalg.inv(rho)
        
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
            z = mu / sigma_vector
            chi2 = rho_inverse.dot(z)
            chi2 = z.dot(chi2)
            return -1*chi2

        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
            
        self.variables = [logp,alpha,pi,mu,prior_pops,self.prior_dirichlet]
        self.pi, self.prior_pops, self.alpha = pi , prior_pops, alpha


        
class Jeffreys_LVBP(LVBP):
    def __init__(self,f_sim,f_exp,sigma_vector,bootstrap_index_list,save_pops=False,alpha0=None,uniform_prior_pops=True,weights_alpha=None):

        LVBP.__init__(self,f_sim,f_exp,sigma_vector,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        alpha = pymc.Uninformative("alpha",value=np.zeros(self.n))
        
        if weights_alpha == None:
            weights0 = pymc.Dirichlet("incomplete_weights",np.ones(self.n) * weights_alpha,value = np.ones(self.n - 1),observed=True)
            weights = pymc.CompletedDirichlet("weights",weights0)
        else:
            weights0 = pymc.Dirichlet("incomplete_weights",np.ones(self.n) * weights_alpha)
            weights = pymc.CompletedDirichlet("weights",weights0)
        
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
        def logp(pi=pi,mu=mu,weights=weights):
            return -1*np.linalg.norm(weights*mu / sigma_vector)**2.

        @pymc.potential
        def jeff(pi=pi,mu=mu):
            return log_jeffreys(pi,f_sim,mu=mu)
        
        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
                    
        self.variables = [logp,alpha,pi,mu,prior_pops,self.prior_dirichlet,jeff,weights0,weights]
        self.pi, self.prior_pops, self.alpha = pi , prior_pops, alpha
        

def dirichlet_to_prior_pops(dirichlet,bootstrap_index_list,m):
    x = np.ones(m)

    pops = np.zeros(len(bootstrap_index_list))
    pops[:-1] = dirichlet[:]
    pops[-1] = 1.0 - pops.sum()

    for k,ind in enumerate(bootstrap_index_list):
        x[ind] = pops[k] / len(ind)
    
    return x
    
def cross_validated_mcmc(f_sim,f_exp,sigma_vector,regularization_strength,bootstrap_index_list,num_samples = 50000,thin=1,model="maxent"):
    all_indices = np.concatenate(bootstrap_index_list)
    test_chi = []
    train_chi = []
    precision = np.cov(f_sim.T)
    for j, train_ind in enumerate(bootstrap_index_list):
        test_ind = np.setdiff1d(all_indices,train_ind)
        num_local_block = 2
        local_bootstrap_index_list = np.array_split(np.arange(len(train_ind)),num_local_block)
        if model == "gaussian":
            S = Gaussian_LVBP(f_sim[train_ind],f_exp,sigma_vector,regularization_strength,local_bootstrap_index_list,precision=precision,uniform_prior_pops=True)
        elif model == "maxent":
            S = MaxEnt_LVBP(f_sim[train_ind],f_exp,sigma_vector,regularization_strength,local_bootstrap_index_list,precision=precision,uniform_prior_pops=True)
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