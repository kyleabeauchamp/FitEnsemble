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
import scipy.io,scipy.optimize, scipy.misc,scipy.linalg,scipy.sparse
import pymc

def get_prior_pops(num_frames, prior_pops=None):
    """Returns a uniform population vector if prior_pops is None.

    Parameters
    ----------
    num_frames : int
        Number of conformations
    prior_pops : ndarray, shape = (num_frames)
        Prior populations of each conformation.  If None, then use uniform pops.   

    Returns
    -------
    prior_pops : ndarray, shape = (num_frames)
        Prior populations of each conformation
    """
    
    if prior_pops != None:
        return prior_pops
    else:
        prior_pops = np.ones(num_frames)
        prior_pops /= prior_pops.sum()
        return prior_pops

def get_populations(alpha, predictions, prior_pops=None):
    """Return the reweighted conformational populations.

    Parameters
    ----------
    alpha : ndarray, shape = (num_measurements)
        Biasing weights for each experiment.
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    prior_pops : ndarray, shape = (num_frames)
        Prior populations of each conformation.  If None, then use uniform pops.       

    Returns
    -------
    populations : ndarray, shape = (num_frames)
        Reweighted populations of each conformation

    """
    num_frames = predictions.shape[0]
    prior_pops = get_prior_pops(num_frames,prior_pops)
    
    q = -1 * predictions.dot(alpha)
    q -= q.mean()  # Improves numerical stability without changing populations.
    populations = np.exp(q)
    populations *= prior_pops
    populations_sum = populations.sum()

    if np.isnan(populations_sum) or np.isinf(populations_sum):  # If we have NAN or inf issues, pick the frame with highest population.
        populations = np.zeros(num_frames)
        populations[q.argmax()] = 1.
    else:
        populations /= populations_sum

    return populations_sum
    
def get_chi2(populations, predictions, measurements, uncertainties, mu=None):
    """Return the chi squared objective function.

    Parameters
    ----------
    alpha : ndarray, shape = (num_measurements)
        Biasing weights for each experiment.
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    prior_pops : ndarray, shape = (num_frames)
        Prior populations of each conformation.  If None, then use uniform pops.       

    Returns
    -------
    populations : ndarray, shape = (num_frames)
        Reweighted populations of each conformation

    """

    if mu == None:
        mu = predictions.T.dot(populations)
    
    delta = (measurements - mu) / uncertainties

    return np.linalg.norm(delta)**2.

def sample_prior_pops(num_frames, bootstrap_index_list):
    """Sample the prior populations using a Dirchlet random variable.

    Parameters
    ----------
    num_frames : int
        Number of conformations
    bootstrap_index_list : list([ndarray])
        List of arrays of frame indices.  The indices in bootstrap_index_list[i]
        will be perturbed together

    Returns
    -------
    prior_populations : ndarray, shape = (num_frames)
        Prior populations of each conformation
        
    Notes
    -------
    This function allows you to perform Bayesian bootstrapping by modifying 
    the prior populations attached to each frame.  Because molecular dynamics
    frames are time correlated, one must first divide the dataset into
    temporal blocks.  A dirichlet random variable is then drawn to modify the prior
    populations blockwise.

    """
    num_blocks = len(bootstrap_index_list)

    prior_dirichlet = pymc.Dirichlet("prior_dirichlet", np.ones(num_blocks))  # Draw a dirichlet

    block_pops = np.zeros(num_blocks)
    block_pops[:-1] = prior_dirichlet[:]  # The pymc Dirichlet does not explicitly store the final component
    block_pops[-1] = 1.0 - block_pops.sum()  # Calculate the final component from normalization.

    prior_populations = np.ones(num_frames)
    for k,ind in enumerate(bootstrap_index_list):
        prior_populations[ind] = block_pops[k] / len(ind)
    
    return prior_populations        

class LVBP():
    """Abstract base class for Linear Virtual Biasing Potential."""
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, predictions, measurements, uncertainties, prior_pops=None):
        """Abstract base class for Linear Virtual Biasing Potential.

        Parameters
        ----------
        predictions : ndarray, shape = (num_frames, num_measurements)
            predictions[j, i] gives the ith observabled predicted at frame j
        measurements : ndarray, shape = (num_measurements)
            measurements[i] gives the ith experimental measurement
        uncertainties : ndarray, shape = (num_measurements)
            uncertainties[i] gives the uncertainty of the ith experiment
        prior_pops : ndarray, shape = (num_frames)
            Prior populations of each conformation.  If None, use uniform populations.        
        """
        self.uncertainties = uncertainties
        self.predictions = predictions
        self.measurements = measurements        
        
        self.num_frames, self.num_measurements = predictions.shape
                          
        if prior_pops == None:
            self.prior_pops = np.ones(self.num_frames) / float(self.num_frames)
        else:
            self.prior_pops = prior_pops
            
    def initialize_variables(self):
        """Must be called by any subclass of LVBP; initializes MCMC variables."""        
        @pymc.dtrm
        def populations(alpha=self.alpha,prior_pops=self.prior_pops):
            return get_populations(alpha, self.predictions, prior_pops)        
        self.populations = populations
        
        @pymc.dtrm
        def mu(populations=self.populations):
            return populations.dot(self.predictions)
        self.mu = mu

        @pymc.potential
        def logp(populations=self.populations,mu=self.mu):
            return -1 * get_chi2(populations, self.predictions, self.measurements, self.uncertainties,mu=mu)
        self.logp = logp
                
    def sample(self, num_samples, thin=1, burn=0,save_pops=False,filename = None):
        """Construct MCMC object and begin sampling."""
        if save_pops == False:
            self.populations.keep_trace = False
        
        if filename == None:
            db = "ram"
        else:
            db = "hdf5"
            
        self.S = pymc.MCMC(self,db=db, dbname=filename)
        self.S.sample(num_samples, thin=thin, burn=burn)
        
    def accumulate_populations(self):
        """Accumulate populations and RMS error over MCMC trace.

        Returns
        -------
        p : ndarray, shape = (num_frames)
            Maximum a posteriori populations of each conformation
        rms: float
            RMS error of model over MCMC trace.        
        """
        a0 = self.S.trace("alpha")[:]        
        p = np.zeros(self.num_frames)
        rms_trace = []
        for i, a in enumerate(a0):
            populations = get_populations(a, self.predictions, self.prior_pops)
            p += populations
            rms_trace.append((get_chi2(populations, self.predictions, self.measurements, self.uncertainties) / float(self.num_measurements)) ** 0.5)
            
        p /= p.sum()
        rms = np.mean(rms_trace)

        return p, rms

class MVN_LVBP(LVBP):
    """Linear Virtual Biasing Potential with MultiVariate Normal Prior."""

    def __init__(self, predictions, measurements, uncertainties, regularization_strength, precision=None, prior_pops=None):
        """Linear Virtual Biasing Potential with MultiVariate Normal Prior.

        Parameters
        ----------
        predictions : ndarray, shape = (num_frames, num_measurements)
            predictions[j, i] gives the ith observabled predicted at frame j
        measurements : ndarray, shape = (num_measurements)
            measurements[i] gives the ith experimental measurement
        uncertainties : ndarray, shape = (num_measurements)
            uncertainties[i] gives the uncertainty of the ith experiment
        regularization_strength : float
            How strongly to weight the MVN prior (e.g. lambda)
        precision : ndarray, optional, shape = (num_measurements, num_measurements)
            The precision matrix of the predicted observables.
        prior_pops : ndarray, optional, shape = (num_frames)
            Prior populations of each conformation.  If None, use uniform populations.        
        """
        LVBP.__init__(self,predictions,measurements,uncertainties,prior_pops=prior_pops)
        
        if precision == None:
            precision = np.cov(predictions.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))

        self.alpha = pymc.MvNormal("alpha", np.zeros(self.num_measurements), tau=precision * regularization_strength)
        self.initialize_variables()

class MaxEnt_LVBP(LVBP):
    """Linear Virtual Biasing Potential with maximum entropy prior."""
    def __init__(self,predictions,measurements,uncertainties,regularization_strength,prior_pops=None):
        """Linear Virtual Biasing Potential with maximum entropy prior.

        Parameters
        ----------
        predictions : ndarray, shape = (num_frames, num_measurements)
            predictions[j, i] gives the ith observabled predicted at frame j
        measurements : ndarray, shape = (num_measurements)
            measurements[i] gives the ith experimental measurement
        uncertainties : ndarray, shape = (num_measurements)
            uncertainties[i] gives the uncertainty of the ith experiment
        regularization_strength : float
            How strongly to weight the MVN prior (e.g. lambda)
        precision : ndarray, optional, shape = (num_measurements, num_measurements)
            The precision matrix of the predicted observables.
        prior_pops : ndarray, optional, shape = (num_frames)
            Prior populations of each conformation.  If None, use uniform populations.        
        """

        LVBP.__init__(self,predictions,measurements,uncertainties,prior_pops=prior_pops)
        
        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))  # The prior on alpha is defined as a potential, so we use Uninformative variables here.
        self.initialize_variables()
        
        @pymc.potential
        def logp_prior(populations=self.populations, mu=self.mu, prior_pops=self.prior_pops):
            if populations.min() <= 0:
                return -1 * np.inf
            else:
                return -1 * regularization_strength * (populations * (np.log(populations / prior_pops))).sum()
        self.logp_prior = logp_prior


class Jeffreys_LVBP(LVBP):
    """Linear Virtual Biasing Potential with Jeffrey's prior."""
    def __init__(self, predictions, measurements, uncertainties, uniform_prior_pops=True, weights_alpha=None):
        """Linear Virtual Biasing Potential with Jeffrey's prior.

        Parameters
        ----------
        predictions : ndarray, shape = (num_frames, num_measurements)
            predictions[j, i] gives the ith observabled predicted at frame j
        measurements : ndarray, shape = (num_measurements)
            measurements[i] gives the ith experimental measurement
        uncertainties : ndarray, shape = (num_measurements)
            uncertainties[i] gives the uncertainty of the ith experiment
        prior_pops : ndarray, optional, shape = (num_frames)
            Prior populations of each conformation.  If None, use uniform populations.        
        """
        LVBP.__init__(self, predictions, measurements, uncertainties, uniform_prior_pops=uniform_prior_pops)

        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))
        self.initialize_variables()
                
        @pymc.potential
        def logp_prior(populations=self.populations,mu=self.mu):
            return log_jeffreys(populations,predictions,mu=mu)
        self.logp_prior = logp_prior

class MaxEnt_Correlation_Corrected_LVBP(LVBP):
    """Linear Virtual Biasing Potential with maximum entropy prior and correlation-corrected likelihood."""
    def __init__(self, predictions, measurements, uncertainties, regularization_strength, precision=None, prior_pops=None):
        """Linear Virtual Biasing Potential with maximum entropy prior and correlation-corrected likelihood.

        Parameters
        ----------
        predictions : ndarray, shape = (num_frames, num_measurements)
            predictions[j, i] gives the ith observabled predicted at frame j
        measurements : ndarray, shape = (num_measurements)
            measurements[i] gives the ith experimental measurement
        uncertainties : ndarray, shape = (num_measurements)
            uncertainties[i] gives the uncertainty of the ith experiment
        regularization_strength : float
            How strongly to weight the MVN prior (e.g. lambda)
        precision : ndarray, optional, shape = (num_measurements, num_measurements)
            The precision matrix of the predicted observables.
        prior_pops : ndarray, optional, shape = (num_frames)
            Prior populations of each conformation.  If None, use uniform populations.        
        """

        LVBP.__init__(self, predictions, measurements, uncertainties, prior_pops=prior_pops)

        if precision == None:
            precision = np.cov(predictions.T)
            if precision.ndim == 0:
                precision = precision.reshape((1,1))        
        
        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))  # The prior on alpha is defined as a potential, so we use Uninformative variables here.
        self.initialize_variables()
        
        @pymc.potential
        def logp_prior(populations=self.populations, mu=self.mu, prior_pops=self.prior_pops):
            if populations.min() <= 0:
                return -1 * np.inf
            else:
                return -1 * regularization_strength * (populations * (np.log(populations / prior_pops))).sum()
        self.logp_prior = logp_prior

        rho = np.corrcoef(predictions.T)
        rho_inverse = np.linalg.inv(rho)
        
        @pymc.potential
        def logp(populations=self.populations,mu=self.mu):
            z = (mu - measurements) / uncertainties
            chi2 = rho_inverse.dot(z)
            chi2 = z.dot(chi2)
            return -1 * chi2
        self.logp = logp
        
    
def cross_validated_mcmc(predictions, measurements, uncertainties, regularization_strength, bootstrap_index_list, num_samples = 50000, thin=1, prior="maxent"):
    all_indices = np.concatenate(bootstrap_index_list)
    test_chi = []
    train_chi = []
    precision = np.cov(predictions.T)
    for j, test_ind in enumerate(bootstrap_index_list):  # The test indices are input as the kfold splits of the data.
        train_ind = np.setdiff1d(all_indices,test_ind)  # The train data is ALL the rest of the data.  Thus, train > test.
        test_data = predictions[test_ind].copy()
        train_data = predictions[train_ind].copy()        

        if prior == "gaussian":
            S = MVN_LVBP(train_data,measurements,uncertainties,regularization_strength,precision=precision,uniform_prior_pops=True)
        elif prior == "maxent":
            S = MaxEnt_LVBP(train_data,measurements,uncertainties,regularization_strength,precision=precision,uniform_prior_pops=True)

        S.test_chi_observable = pymc.Lambda("test_chi_observable", 
                                          lambda alpha=S.alpha: get_chi2(get_populations(alpha, test_data), test_data, measurements, uncertainties))
        S.train_chi_observable = pymc.Lambda("train_chi_observable", 
                                           lambda populations=S.populations,mu=S.mu: get_chi2(populations, train_data, measurements, uncertainties, mu=mu))
                                           
        S.sample(num_samples,thin=thin)
        test_chi.append(S.S.trace("test_chi_observable")[:].mean())
        train_chi.append(S.S.trace("train_chi_observable")[:].mean())

    test_chi = np.array(test_chi)
    train_chi = np.array(train_chi)
    print regularization_strength, train_chi.mean(), test_chi.mean()
    return train_chi, test_chi
            
def fisher_information(populations, predictions, mu):
    """Calculate the fisher information.
    
    Parameters
    ----------
    populations : ndarray, shape = (num_frames)
        populations of each frame
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    mu : ndarray, shape = (num_measurements)
        Ensemble average prediction of each observable.        
    """
    D = predictions - mu
    num_frames = len(populations)
    Dia = scipy.sparse.dia_matrix((populations,0),(num_frames, num_frames))
    D = Dia.dot(D)
    S = D.T.dot(predictions)
    return S

def log_jeffreys(populations, predictions, mu):
    """Calculate the log of Jeffrey's prior.
    
    Parameters
    ----------
    populations : ndarray, shape = (num_frames)
        populations of each frame
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    mu : ndarray, shape = (num_measurements)
        Ensemble average prediction of each observable.        
    """

    I = fisher_information(populations,predictions,mu)
    sign,logdet = np.linalg.slogdet(I)
    return 0.5 * logdet