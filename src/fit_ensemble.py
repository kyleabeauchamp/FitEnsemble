"""

Most functions in this module take three inputs: alpha, predictions, and measurements.
All these functions share a common model framework.  We have run a simulation
and calculated a set of observables at each conformation.  Thus, predictions[j,i]
gives the ith predicted observable at frame j.  We assume that our 
experimental data consists of n equilibrium measurements--the ith experiment
is stored in measurements[i].  Finally, alpha is a vector of coupling parameters 
that will be used to reweight the simulation value using a biasing potential 
that is linear in the predicted observables: 
U(x_i) = \sum_j alpha[i] predictions[j,i]

"""
import abc
import numpy as np
import scipy.stats, scipy.io,scipy.optimize, scipy.misc,scipy.linalg,scipy.sparse
import pymc

def get_prior_pops(num_frames, prior_pops=None):
    """Returns uniform distribution if prior_pops is None."""
    if prior_pops != None:
        return prior_pops
    else:
        x = np.ones(num_frames)
        x /= x.sum()
        return x

def populations(alpha, predictions, prior_pops=None):
    """Return the reweighted conformational populations."""

    prior_pops = get_prior_pops(len(predictions),prior_pops)
    
    q = -1 * predictions.dot(alpha)
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
    
def chi2(pi, predictions, measurements, uncertainties, mu=None):
    """Return the chi squared objective function.

    Notes
    -----

    References
    ----------
    .. [1] Beauchamp, K. A. 
    """

    if mu == None:
        mu = predictions.T.dot(pi)
    
    delta = (measurements - mu) / uncertainties

    return np.linalg.norm(delta)**2.
        

class LVBP():
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,predictions,measurements,uncertainties,bootstrap_index_list,alpha0=None,uniform_prior_pops=True):
        self.uncertainties = uncertainties
        self.prior_pops = None
        self.predictions = predictions
        self.measurements = measurements        
        self.bootstrap_index_list = bootstrap_index_list
        
        self.num_frames, self.num_measurements = predictions.shape
                  
        if alpha0 == None:
            alpha0 = np.zeros(self.num_measurements)
        
        if uniform_prior_pops == True:
            self.prior_dirichlet = pymc.Dirichlet("prior_dirichlet", np.ones(len(self.bootstrap_index_list)), value=np.ones(len(bootstrap_index_list) - 1) / float(len(bootstrap_index_list)))
        else:
            self.prior_dirichlet = pymc.Dirichlet("prior_dirichlet", np.ones(len(self.bootstrap_index_list)))

        self.prior_dirichlet = pymc.Dirichlet("prior_dirichlet", np.ones(len(self.bootstrap_index_list)), value=self.prior_dirichlet.value, observed=True)


    def initialize_deterministics(self):
        """This must be called by any subclass of LVBP."""
        @pymc.dtrm
        def prior_pops(prior_dirichlet=self.prior_dirichlet):
            return dirichlet_to_prior_pops(prior_dirichlet,self.bootstrap_index_list,len(self.predictions))
        self.prior_pops = prior_pops
        
        @pymc.dtrm
        def pi(alpha=self.alpha,prior_pops=self.prior_pops):
            return populations(alpha, self.predictions, prior_pops)        
        self.pi = pi
        
        @pymc.dtrm
        def mu(pi=self.pi):
            return pi.dot(self.predictions)
        self.mu = mu

    def initialize_potentials(self):
        @pymc.potential
        def logp(pi=self.pi,mu=self.mu):
            return -1 * chi2(pi, self.predictions, self.measurements, self.uncertainties,mu=mu)
        self.logp = logp
    
    def initialize_variables(self):
        self.initialize_deterministics()
        self.initialize_potentials()
            
    def sample(self, num_samples, thin=1, burn=0,save_pops=False,filename = None):
        """Construct MCMC object and begin sampling."""
        if save_pops == False:
            self.pi.keep_trace = False
            self.prior_pops.keep_trace = False
        
        if filename == None:
            db = "ram"
        else:
            db = "hdf5"
            
        self.S = pymc.MCMC(self,db=db, dbname=filename)
        self.S.sample(num_samples, thin=thin, burn=burn)
        
    def accumulate_populations(self):
        """Accumulate populations and RMS error over MCMC trace."""
        a0 = self.S.trace("alpha")[:]        
        p = np.zeros(self.num_frames)
        rms_trace = []
        for i,a in enumerate(a0):
            pi = populations(a, self.predictions, self.prior_pops.value)
            p += pi
            rms_trace.append((chi2(pi, self.predictions, self.measurements, self.uncertainties) / float(self.num_measurements)) ** 0.5)
            
        p /= p.sum()
        rms = np.mean(rms_trace)

        return p, rms

class Gaussian_LVBP(LVBP):

    def __init__(self, predictions, measurements, uncertainties, regularization_strength, bootstrap_index_list, precision=None, alpha0=None, uniform_prior_pops=True):

        LVBP.__init__(self,predictions,measurements,uncertainties,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(predictions.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))

        self.alpha = pymc.MvNormal("alpha", np.zeros(self.num_measurements), tau=precision * regularization_strength, value=alpha0)
        self.initialize_variables()

class MaxEnt_LVBP(LVBP):

    def __init__(self,predictions,measurements,uncertainties,regularization_strength,bootstrap_index_list,precision=None,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,predictions,measurements,uncertainties,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(predictions.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))

        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))
        self.initialize_variables()
        
        @pymc.potential
        def logp_prior(pi=self.pi,mu=self.mu):
            if pi.min() <= 0:
                return -1 * np.inf
            else:
                return -1 * regularization_strength * sum((pi * np.log(pi)))
        self.logp_prior = logp_prior


class Gaussian_Corr_LVBP(LVBP):

    def __init__(self,predictions,measurements,uncertainties,regularization_strength,bootstrap_index_list,precision=None,alpha0=None,uniform_prior_pops=True):

        LVBP.__init__(self,predictions,measurements,uncertainties,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        
        if precision == None:
            precision = np.cov(predictions.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))
        
        self.alpha = pymc.MvNormal("alpha", np.zeros(self.num_measurements), tau=precision * regularization_strength, value=alpha0)
        self.initialize_variables()

        rho = np.corrcoef(predictions.T)
        rho_inverse = np.linalg.inv(rho)
        
        @pymc.potential
        def logp(pi=self.pi,mu=self.mu):
            z = (mu - measurements) / uncertainties
            chi2 = rho_inverse.dot(z)
            chi2 = z.dot(chi2)
            return -1 * chi2
        self.logp = logp            
        
class Jeffreys_LVBP(LVBP):
    def __init__(self,predictions,measurements,uncertainties,bootstrap_index_list,alpha0=None,uniform_prior_pops=True,weights_alpha=None):

        LVBP.__init__(self,predictions,measurements,uncertainties,bootstrap_index_list,alpha0=alpha0,uniform_prior_pops=uniform_prior_pops)
        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))
        self.initialize_variables()
                
        @pymc.potential
        def logp_prior(pi=self.pi,mu=self.mu):
            return log_jeffreys(pi,predictions,mu=mu)
        self.logp_prior = logp_prior
        

def dirichlet_to_prior_pops(dirichlet,bootstrap_index_list,num_frames):
    x = np.ones(num_frames)

    pops = np.zeros(len(bootstrap_index_list))
    pops[:-1] = dirichlet[:]
    pops[-1] = 1.0 - pops.sum()

    for k,ind in enumerate(bootstrap_index_list):
        x[ind] = pops[k] / len(ind)
    
    return x
    
def cross_validated_mcmc(predictions,measurements,uncertainties,regularization_strength,bootstrap_index_list,num_samples = 50000,thin=1,model="maxent"):
    all_indices = np.concatenate(bootstrap_index_list)
    test_chi = []
    train_chi = []
    precision = np.cov(predictions.T)
    for j, train_ind in enumerate(bootstrap_index_list):
        test_ind = np.setdiff1d(all_indices,train_ind)
        num_local_block = 2
        local_bootstrap_index_list = np.array_split(np.arange(len(train_ind)),num_local_block)
        if model == "gaussian":
            S = Gaussian_LVBP(predictions[train_ind],measurements,uncertainties,regularization_strength,local_bootstrap_index_list,precision=precision,uniform_prior_pops=True)
        elif model == "maxent":
            S = MaxEnt_LVBP(predictions[train_ind],measurements,uncertainties,regularization_strength,local_bootstrap_index_list,precision=precision,uniform_prior_pops=True)
        test_chi_observable = pymc.Lambda("test_chi_observable",lambda alpha=S.alpha: chi2(alpha,predictions[test_ind],measurements))
        train_chi_observable = pymc.Lambda("train_chi_observable",lambda alpha=S.alpha: chi2(alpha,predictions[train_ind],measurements))
        S.variables.append(test_chi_observable)
        S.variables.append(train_chi_observable)
        S.sample(num_samples,thin=thin)
        test_chi.append(S.S.trace("test_chi_observable")[:].mean())
        train_chi.append(S.S.trace("train_chi_observable")[:].mean())

    test_chi = np.array(test_chi)
    train_chi = np.array(train_chi)
    print regularization_strength, train_chi.mean(), test_chi.mean()
    return test_chi,train_chi
            
def fisher_information(populations,predictions,mu):
    D = predictions - mu
    num_frames = len(populations)
    Dia = scipy.sparse.dia_matrix((populations,0),(num_frames, num_frames))
    D = Dia.dot(D)
    S = D.T.dot(predictions)
    return S

def log_jeffreys(populations,predictions,mu):
    I = fisher_information(populations,predictions,mu)
    sign,logdet = np.linalg.slogdet(I)
    return 0.5 * logdet