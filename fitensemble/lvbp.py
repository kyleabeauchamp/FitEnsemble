"""
An implementation of the LVBP approach outlined in Beauchamp, JACS. 2013.  

Most functions in this module take three inputs: alpha, predictions, and measurements.
Thus, predictions[j,i] gives the ith predicted observable at frame j.  The ith 
equilibrium experiment is stored in measurements[i].  Similarly, uncertainties[i]
contains an estimate of the uncertainty in the ith measurement.  In practice, we 
use uncertainties[i] to model *both* the experimental uncertainty and the 
prediction uncertainty associated with predicting experimental observables.  

Below, alpha is a vector of coupling parameters that will be used to reweight
 the simulation value using a biasing potential that is linear in the predicted observables: 
U(x_i) = \sum_j alpha[i] predictions[j,i]

"""
import abc
import numpy as np
import scipy.sparse
import pymc
from ensemble_fitter import EnsembleFitter, get_prior_pops, get_chi2
import numexpr as ne

ne.set_num_threads(3)  # I found optimal performance for 3 threads.  Also you should use OMP_NUM_THREADS=2

def get_q(alpha, predictions):
    """Project predictions onto alpha.

    Parameters
    ----------
    alpha : ndarray, shape = (num_measurements)
        Biasing weights for each experiment.
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j

    Returns
    -------
    q : ndarray, shape = (num_frames)
        q is -1.0 * predictions.dot(alpha)
    Notes
    -----
    We also mean-subtract q, as this provides improved 
    numerical stability.  
    """
    q = predictions.dot(-1.0 * alpha)
    q -= q.mean()  # Improves numerical stability without changing populations.
    return q

def get_populations_from_q(q, predictions, prior_pops):
    """Return the reweighted conformational populations.

    Parameters
    ----------
    q : ndarray, shape = (num_frames)
        q is -1.0 * predictions.dot(alpha)
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    prior_pops : ndarray, shape = (num_frames)
        Prior populations of each conformation.  If None, then use uniform pops.       

    Returns
    -------
    populations : ndarray, shape = (num_frames)
        Reweighted populations of each conformation

    Notes
    -----
    We typically input the mean-subtracted q, as this provides improved 
    numerical stability.  
    """
    populations = ne.evaluate("exp(q) * prior_pops")
    populations_sum = populations.sum()

    if np.isfinite(populations_sum):
        populations /= populations_sum
    else:  # If we have NAN or inf issues, pick the frame with highest population.
        populations.fill(0.0)
        populations[q.argmax()] = 1.

    return populations

def get_populations_from_alpha(alpha, predictions, prior_pops):
    """Return the reweighted conformational populations.

    Parameters
    ----------
    alpha : ndarray, shape = (num_measurements)
        Biasing weights for each experiment.
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    prior_pops : ndarray, shape = (num_frames)
        Prior populations of each conformation. 

    Returns
    -------
    populations : ndarray, shape = (num_frames)
        Reweighted populations of each conformation

    """
    q = get_q(alpha, predictions)
    return get_populations_from_q(q, predictions, prior_pops)    

class LVBP(EnsembleFitter):
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
        EnsembleFitter.__init__(self, predictions, measurements, uncertainties, prior_pops=prior_pops)
            
    def initialize_variables(self):
        """Must be called by any subclass of LVBP; initializes MCMC variables."""        

        @pymc.dtrm
        def q(alpha=self.alpha, prior_pops=self.prior_pops):
            return get_q(alpha, self.predictions)
        self.q = q
        self.q.keep_trace = False            

        @pymc.dtrm
        def populations(q=self.q, prior_pops=self.prior_pops):
            return get_populations_from_q(q, self.predictions, prior_pops)        
        self.populations = populations
        self.populations.keep_trace = False        
        
        @pymc.dtrm
        def mu(populations=self.populations):
            return populations.dot(self.predictions)
        self.mu = mu

        @pymc.potential
        def logp(populations=self.populations,mu=self.mu):
            return -1 * get_chi2(populations, self.predictions, self.measurements, self.uncertainties, mu=mu)
        self.logp = logp

    def iterate_populations(self):
        alpha_trace = self.mcmc.trace("alpha")[:]  # Assume we can load *all* alpha into memory.  I.e. num_measurements small.
        for i, alpha in enumerate(alpha_trace):
            populations = get_populations_from_alpha(alpha, self.predictions, self.prior_pops)
            yield populations
        
class MVN_LVBP(LVBP):
    """Linear Virtual Biasing Potential with MultiVariate Normal Prior."""

    def __init__(self, predictions, measurements, uncertainties, regularization_strength=1.0, precision=None, prior_pops=None):
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
        LVBP.__init__(self, predictions, measurements, uncertainties, prior_pops=prior_pops)
        
        if precision == None:
            precision = np.cov(predictions.T)
            if precision.ndim == 0:
                precision = precision.reshape((1, 1))

        self.alpha = pymc.MvNormal("alpha", np.zeros(self.num_measurements), tau=precision * regularization_strength)
        self.initialize_variables()

class MaxEnt_LVBP(LVBP):
    """Linear Virtual Biasing Potential with maximum entropy prior."""
    def __init__(self, predictions, measurements, uncertainties, regularization_strength=1.0, prior_pops=None):
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

        self.log_prior_pops = np.log(self.prior_pops)
        
        @pymc.potential
        def logp_prior(populations=self.populations, q=self.q, log_prior_pops=self.log_prior_pops):
            if populations.min() <= 0:
                return -1 * np.inf
            else:
                expr = populations.dot(q) - populations.dot(log_prior_pops)
                return -1 * regularization_strength * expr
        self.logp_prior = logp_prior


class Jeffreys_LVBP(LVBP):
    """Linear Virtual Biasing Potential with Jeffrey's prior."""
    def __init__(self, predictions, measurements, uncertainties, prior_pops=None, weights_alpha=None):
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
        LVBP.__init__(self, predictions, measurements, uncertainties, prior_pops=prior_pops)

        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))
        self.initialize_variables()
                
        @pymc.potential
        def logp_prior(populations=self.populations,mu=self.mu):
            return log_jeffreys(populations,predictions,mu=mu)
        self.logp_prior = logp_prior

class MaxEnt_Correlation_Corrected_LVBP(LVBP):
    """Linear Virtual Biasing Potential with maximum entropy prior and correlation-corrected likelihood."""
    def __init__(self, predictions, measurements, uncertainties, regularization_strength=1.0, precision=None, prior_pops=None):
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
        
    
def cross_validated_mcmc(predictions, measurements, uncertainties, model_factory, bootstrap_index_list, num_samples = 50000, thin=1, prior="maxent"):
    all_indices = np.concatenate(bootstrap_index_list)
    test_chi = []
    train_chi = []
    precision = np.cov(predictions.T)
    
    for j, test_ind in enumerate(bootstrap_index_list):  # The test indices are input as the kfold splits of the data.
        train_ind = np.setdiff1d(all_indices,test_ind)  # The train data is ALL the rest of the data.  Thus, train > test.
        test_data = predictions[test_ind].copy()
        train_data = predictions[train_ind].copy()        

        test_prior_pops = np.ones_like(test_data[:,0])
        test_prior_pops /= test_prior_pops.sum()
        
        print("Building model for %d round of cross validation." % j)
        model = model_factory(train_data, measurements, uncertainties)
        model.sample(num_samples,thin=thin)
        
        train_chi2_j = []  # Calculate the chi2 error on training data
        for k, alpha in enumerate(model.mcmc.trace("alpha")):
            p = get_populations_from_alpha(alpha, train_data, model.prior_pops)  # Training set prior_pops has correct shape
            chi2 = get_chi2(p, train_data, measurements, uncertainties)
            train_chi2_j.append(chi2)

        test_chi2_j = []  # Calculate the chi2 error on test data
        for k, alpha in enumerate(model.mcmc.trace("alpha")):
            p = get_populations_from_alpha(alpha, test_data, test_prior_pops)  # Training set prior_pops has correct shape
            chi2 = get_chi2(p, test_data, measurements, uncertainties)
            test_chi2_j.append(chi2)        

        test_chi.append(np.mean(test_chi2_j))
        train_chi.append(np.mean(train_chi2_j))

    test_chi = np.array(test_chi)
    train_chi = np.array(train_chi)
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
