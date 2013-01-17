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
import scipy.stats, scipy.io,scipy.optimize, scipy.misc,scipy.linalg,scipy.sparse
import pymc

def automatic_subsampling_selection(f_sim):
    x = (f_sim - f_sim.mean(0))
    n = f_sim.shape[1]
    cv = np.cov(f_sim.T)
    cr = np.array([[np.mean(x[:-1,i]*x[1:,j]) for i in xrange(n)] for j in xrange(n)])
    
    lam, ev =  scipy.linalg.eig(cr,b=cv)
    tau = -1 / np.log(lam.max())
    
    tau *= 0.51
    tau = int(tau)

    print("Automatically subsampling data by %d"%tau)
    

    return tau

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
    
    

def sample_mcmc_old(f_sim,f_exp,regularization_strength,num_samples,thin=4):
    n = len(f_exp)
    alpha0 = np.zeros(n)
    alpha0 = minimize_chi2(alpha0,f_sim,f_exp,regularization_strength)
    
    alpha = pymc.MvNormal("alpha",alpha0,np.eye(n) * regularization_strength,trace=True)
    
    @pymc.deterministic
    def observable(alpha=alpha):
        p = populations(alpha,f_sim,f_exp)
        return p
        
    def f(alpha,observable):
        return -1*chi2(alpha,f_sim,f_exp) - ridge(alpha)*regularization_strength
    
    potential = pymc.Potential(logp = f, doc="doc",name = 'psi', parents = {'alpha': alpha,"observable":observable})
    
    S = pymc.MCMC(potential)
    S.sample(iter=num_samples, burn=1000, thin=thin)
    return S

class MCMC_Sampler():
    def __init__(self,f_sim,f_exp,regularization_strength,bootstrap_index_list,precision=None,save_pops=False,alpha0=None,uniform_prior_pops=True):
        D = f_sim - f_exp
        self.prior_pops = None
        self.f_sim = f_sim
        self.f_exp = f_exp
        self.D = D
        
        self.bootstrap_index_list = bootstrap_index_list
        
        m,n = f_sim.shape
        if precision == None:
            precision = np.cov(f_sim.T)
                  
        if alpha0 == None:
            alpha0 = np.zeros(n)

        alpha = pymc.MvNormal("alpha",np.zeros(n),tau=precision*regularization_strength,value=alpha0)
        
        if uniform_prior_pops == True:
            prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(self.bootstrap_index_list)),value=np.ones(len(bootstrap_index_list) - 1) / float(len(bootstrap_index_list)))
        else:
            prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(self.bootstrap_index_list)))

        prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(self.bootstrap_index_list)),value=prior_dirichlet.value,observed=True)
        
        @pymc.dtrm
        def prior_pops(prior_dirichlet=prior_dirichlet):
            return dirichlet_to_prior_pops(prior_dirichlet,self.bootstrap_index_list,len(self.f_sim))
        
        @pymc.dtrm
        def pi(alpha=alpha,prior_pops=prior_pops):
            return populations(alpha,f_sim,f_exp,prior_pops)
            
        @pymc.potential
        def logp(pi=pi):
            means = pi.dot(D)
            f = -1*np.linalg.norm(means)**2.
            return f
        
        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
        
        self.pi = pi
        self.logp = logp
        self.alpha = alpha
        self.prior_pops = prior_pops
        self.variables = [logp,alpha,pi,prior_pops,prior_dirichlet]
            
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
        



class MCMC_Sampler_BB_OLD():
    def __init__(self,f_sim,f_exp,regularization_strength,bootstrap_index_list,precision=None,save_pops=False):
        D = f_sim - f_exp
        
        m,n = f_sim.shape

        if precision == None:
            precision = np.cov(f_sim.T)

        alpha = pymc.MvNormal("alpha",np.zeros(n),tau=precision*regularization_strength)
        
        prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(bootstrap_index_list)))

        @pymc.dtrm
        def prior_pops(prior_dirchlet=prior_dirichlet):
            return dirichlet_to_prior_pops(prior_dirichlet,bootstrap_index_list,m)
        
        @pymc.dtrm
        def pi(alpha=alpha,prior_pops=prior_pops):
            return populations(alpha,f_sim,f_exp,prior_pops=prior_pops)
            
        @pymc.potential
        def logp(pi=pi):
            means = pi.dot(D)
            f = -1*np.linalg.norm(means)**2.
            print("p[0] = %f, f = %f"%(pi[0],f))
            return f
        
        if save_pops == False:
            pi.keep_trace = False
            prior_pops.keep_trace = False
        
        self.pi = pi
        self.logp = logp
        self.alpha = alpha
        self.variables = [logp,alpha,pi,prior_dirichlet,prior_pops]
        self.initialized = False
        
    def initialize(self):
        self.S = pymc.MCMC(self.variables)
        self.initialized = True
        
    def sample(self,num_samples,thin=1,burn=0):
        if not self.initialized:
            self.initialize()        
        self.S.sample(num_samples,thin=thin,burn=burn)

class MCMC_Sampler_BB():
    def __init__(self,f_sim,f_exp,regularization_strength,bootstrap_index_list,dirichlet_frequency,precision=None,save_pops=False):

        self.bootstrap_index_list = bootstrap_index_list
        D = f_sim - f_exp
        m,n = f_sim.shape
        self.m = m        

        if precision == None:
            precision = np.cov(f_sim.T)

        alpha = pymc.MvNormal("alpha",np.zeros(n),tau=precision*regularization_strength)
        
        self.prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(bootstrap_index_list)))
        self.dirichlet_tracker = []
        self.dirichlet_counter = 0
        self.dirichlet_frequency = dirichlet_frequency
                
        @pymc.dtrm
        def pi(alpha=alpha):
            self.update_prior_pops()
            #self.dirichlet_tracker.append(self.prior_dirichlet.value)
            return populations(alpha,f_sim,f_exp,prior_pops=self.prior_pops)
            
        @pymc.potential
        def logp(pi=pi):
            means = pi.dot(D)
            return -1*np.linalg.norm(means)**2.
        
        if save_pops == False:
            pi.keep_trace = False
        
        self.pi = pi
        self.logp = logp
        self.alpha = alpha
        self.variables = [logp,alpha,pi]
        self.initialized = False
        
    def initialize(self):
        self.S = pymc.MCMC(self.variables)
        self.initialized = True
        
    def sample(self,num_samples,thin=1,burn=0):
        if not self.initialized:
            print("Initializing MCMC")
            self.initialize()
            
        self.S.sample(num_samples,thin=thin,burn=burn)
        self.dirichlet_tracker = np.array(self.dirichlet_tracker)
        
    def update_prior_pops(self):
        if self.dirichlet_counter % self.dirichlet_frequency == 0:
            print("Updating prior probs")
            r = self.prior_dirichlet.rand()
            self.prior_pops = dirichlet_to_prior_pops(self.prior_dirichlet,self.bootstrap_index_list,self.m)
        self.dirichlet_counter += 1

class relent_mcmc_sampler():
    def __init__(self,f_sim,f_exp,regularization_strength,save_pops=False):
        D = f_sim - f_exp
        
        m,n = f_sim.shape
    
        precision = np.eye(n)
        alpha = pymc.MvNormal("alpha",np.zeros(n),tau=precision)
        
        @pymc.dtrm
        def pi(alpha=alpha):
            return populations(alpha,f_sim,f_exp)
            
        @pymc.potential
        def logp(pi=pi):
            means = pi.dot(D)
            return -1*np.linalg.norm(means)**2.

        @pymc.potential
        def relent(pi=pi):
            return -1*regularization_strength*(pi*np.log(pi)).sum()
        
        if save_pops == False:
            pi.keep_trace = False
        
        self.pi = pi
        self.logp = logp
        self.alpha = alpha
        self.variables = [logp,alpha,pi,relent]
        
    def sample(self,num_samples,thin=1):
        self.S = pymc.MCMC(self.variables)
        self.S.sample(num_samples,thin=thin)


class jeffreys_mcmc_sampler():
    def __init__(self,f_sim,f_exp,save_pops=False):
        D = f_sim - f_exp
        
        m,n = f_sim.shape    
        alpha = pymc.Uninformative("alpha",value=np.zeros(n))
        
        @pymc.dtrm
        def pi(alpha=alpha):
            return populations(alpha,f_sim,f_exp)
            
        @pymc.potential
        def logp(pi=pi):
            means = pi.dot(D)
            return -1*np.linalg.norm(means)**2.

        @pymc.potential
        def jeff(pi=pi):
            j = jeffreys(pi,f_sim)
            if j == 0.:
                return -1*np.inf
            return np.log(j)
        
        if save_pops == False:
            pi.keep_trace = False
                    
        self.pi = pi
        self.logp = logp
        self.alpha = alpha
        self.variables = [logp,alpha,pi,jeff]
        
    def sample(self,num_samples,thin=1):
        self.S = pymc.MCMC(self.variables)
        self.S.sample(num_samples,thin=thin)    

def dirichlet_to_prior_pops(dirichlet,bootstrap_index_list,m):
    x = np.ones(m)

    pops = np.zeros(len(bootstrap_index_list))
    pops[:-1] = dirichlet[:]
    pops[-1] = 1.0 - pops.sum()

    for k,ind in enumerate(bootstrap_index_list):
        x[ind] = pops[k] / len(ind)
    
    return x
    
def sample_mcmc_bayesian_bootstrapped(f_sim,f_exp,regularization_strength,num_samples,bootstrap_index_list,thin=4):
    n = len(f_exp)
    m = len(f_sim)
    alpha0 = np.zeros(n)
    alpha0 = minimize_chi2(alpha0,f_sim,f_exp,regularization_strength)
    
    alpha = pymc.MvNormal("alpha",alpha0,np.eye(n) * regularization_strength,trace=True)
    prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(len(bootstrap_index_list)))
            
    def f(alpha,prior_dirichlet):
        prior_pops = dirichlet_to_prior_pops(prior_dirichlet,bootstrap_index_list,m)
        return -1*chi2(alpha,f_sim,f_exp,prior_pops) - ridge(alpha)*regularization_strength
    
    potential = pymc.Potential(logp = f, doc="doc",name = 'psi', parents = {'alpha': alpha,"prior_dirichlet":prior_dirichlet})
    
    S = pymc.MCMC(potential)
    S.sample(iter=num_samples, burn=1000, thin=thin)
    return S    

def cross_validated_mcmc(f_sim,f_exp,regularization_strength,bootstrap_index_list,num_samples = 50000,thin=1):
    all_indices = np.concatenate(bootstrap_index_list)
    test_chi = []
    train_chi = []
    precision = np.cov(f_sim.T)
    for j, train_ind in enumerate(bootstrap_index_list):
        test_ind = np.setdiff1d(all_indices,train_ind)
        num_local_block = 2
        local_bootstrap_index_list = np.array_split(np.arange(len(train_ind)),num_local_block)
        S = MCMC_Sampler(f_sim[train_ind],f_exp,regularization_strength,local_bootstrap_index_list,precision=precision,uniform_prior_pops=True)
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

    
def cross_validated_mcmc_old(f_sim,f_exp,regularization_strength,bootstrap_index_list,num_samples = 50000,thin=1):
    n = len(f_exp)    
    all_indices = np.concatenate(bootstrap_index_list)    
    test_chi = []
    train_chi = []
    for j, train_ind in enumerate(bootstrap_index_list):
        print("CV Round %d"%j)
        test_ind = np.setdiff1d(all_indices,train_ind)
        f_sim_train = f_sim[train_ind].copy()
        f_sim_test = f_sim[test_ind].copy()    
        alpha0 = np.zeros(n)
        alpha0 = minimize_chi2(alpha0,f_sim_train,f_exp,regularization_strength)    
        alpha = pymc.MvNormal("alpha",alpha0,np.eye(n) * regularization_strength, trace=True)    
        f_train = lambda alpha,score_train,score_test: -1*chi2(alpha,f_sim_train,f_exp)    
        @pymc.deterministic
        def score_train(alpha=alpha):
            return chi2(alpha,f_sim_train,f_exp)    
        @pymc.deterministic
        def score_test(alpha=alpha):
            return chi2(alpha,f_sim_test,f_exp)    
        potential = pymc.Potential(logp = f_train, doc="doc",name = 'psi', parents = {'alpha': alpha,"score_train":score_train,"score_test":score_test})        
        S = pymc.MCMC(potential)
        S.sample(iter=num_samples, burn=1000, thin=thin)
        a0 = S.trace("alpha")[:]
        a = a0.mean(0)
        p = populations(a,f_sim,f_exp)
        test_chi.append(S.trace("score_test")[:])
        train_chi.append(S.trace("score_train")[:])
    
    test_chi = np.array(test_chi)
    train_chi = np.array(train_chi)
    print("")
    print regularization_strength,p.max(), train_chi.mean(), test_chi.mean()
    print("chi_test = %f"%test_chi.mean())
    return test_chi,train_chi
    
    
def reconstruct_population_timeseries(f_sim,f_exp,S,bootstrap_index_list):
    alpha_list = S.trace("alpha")[:]
    dirichlet_list = S.trace("prior_dirichlet")[:]
    m,n = f_sim.shape
    num_samples = len(alpha_list)
    p_list = np.zeros((num_samples,m))
    for i,alpha in enumerate(alpha_list):
        prior_dirichlet = dirichlet_list[i]
        p0 = dirichlet_to_prior_pops(prior_dirichlet,bootstrap_index_list,m)
        p_list[i] = populations(alpha,f_sim,f_exp,p0)
    return p_list
    
    
def old_fisher_information(populations,f_sim):
    mu = f_sim.T.dot(populations)
    n = len(populations)
    D = scipy.sparse.dia_matrix((populations,0),(n,n))
    mu_ab = D.dot(f_sim).T.dot(f_sim)
    I = np.outer(mu,mu)
    I -= mu_ab
    return I
    
def fisher_information(populations,f_sim):
    mu = f_sim.T.dot(populations)
    D = f_sim - mu
    n = len(populations)
    Dia = scipy.sparse.dia_matrix((populations,0),(n,n))
    D = Dia.dot(D)
    S = D.T.dot(f_sim)
    return S

def jeffreys(populations,f_sim):
    I = fisher_information(populations,f_sim)
    return np.abs(np.linalg.det(I))**0.5