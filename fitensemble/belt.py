"""
An implementation of the BELT approach outlined in Beauchamp, JACS. 2013.

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
import pymc
from ensemble_fitter import EnsembleFitter, get_chi2
import numexpr as ne
import math

ne.set_num_threads(2)  # I found optimal performance for 2-3 threads.  Also you should use OMP_NUM_THREADS=2  Optimal performance depends on the input size and your system specs.

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
    q = predictions.dot(-1.0 * alpha)
    mu = q.mean()

    if not np.isfinite(mu):  # Sometimes numpy's mean is not numerically stable enough, so we switch to using fsum()
        print("Warning: possible numerical instability detected.")
        q1 = (q / float(len(q)))
        mu = math.fsum(q1)

    q -= mu  # Improves numerical stability without changing populations.
    
    populations = ne.evaluate("exp(q) * prior_pops")
    populations_sum = populations.sum()

    if np.isfinite(populations_sum):
        populations /= populations_sum
    else:  # If we have NAN or inf issues, pick the frame with highest population.
        populations.fill(0.0)
        populations[q.argmax()] = 1.    

    return populations


class BELT(EnsembleFitter):
    """Abstract base class for Bayesian Energy Landscape Tilting."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, predictions, measurements, uncertainties, prior_pops=None):
        """Abstract base class for Bayesian Energy Landscape Tilting.

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
        """Must be called by any subclass of BELT; initializes MCMC variables."""

        @pymc.dtrm
        def populations(alpha=self.alpha, prior_pops=self.prior_pops):
            return get_populations_from_alpha(alpha, self.predictions, prior_pops)
        self.populations = populations
        self.populations.keep_trace = False

        @pymc.dtrm
        def mu(populations=self.populations):
            return populations.dot(self.predictions)
        self.mu = mu

        @pymc.potential
        def logp(populations=self.populations,mu=self.mu):
            return -0.5 * get_chi2(populations, self.predictions, self.measurements, self.uncertainties, mu=mu)
        self.logp = logp

    def iterate_populations(self):
        alpha_trace = self.mcmc.trace("alpha")[:]  # Assume we can load *all* alpha into memory.  I.e. num_measurements small.
        for i, alpha in enumerate(alpha_trace):
            populations = get_populations_from_alpha(alpha, self.predictions, self.prior_pops)
            yield populations


class MVNBELT(BELT):
    """Bayesian Energy Landscape Tilting with MultiVariate Normal Prior."""

    def __init__(self, predictions, measurements, uncertainties, regularization_strength=1.0, precision=None, prior_pops=None):
        """Bayesian Energy Landscape Tilting with MultiVariate Normal Prior.

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
        BELT.__init__(self, predictions, measurements, uncertainties, prior_pops=prior_pops)

        if precision == None:
            precision = np.cov(predictions.T)
            if precision.ndim == 0:
                precision = precision.reshape((1, 1))

        self.alpha = pymc.MvNormal("alpha", np.zeros(self.num_measurements), tau=precision * regularization_strength)
        self.initialize_variables()


class MaxEntBELT(BELT):
    """Bayesian Energy Landscape Tilting with maximum entropy prior."""
    def __init__(self, predictions, measurements, uncertainties, regularization_strength=1.0, prior_pops=None):
        """Bayesian Energy Landscape Tilting with maximum entropy prior.

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

        BELT.__init__(self,predictions,measurements,uncertainties,prior_pops=prior_pops)

        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))  # The prior on alpha is defined as a potential, so we use Uninformative variables here.
        self.initialize_variables()

        self.log_prior_pops = np.log(self.prior_pops)

        @pymc.potential
        def logp_prior(populations=self.populations, log_prior_pops=self.log_prior_pops):
            # So x log(x) -> 0 as x -> 0, so we want to *drop* zeros
            # This is important because we otherwise might get NANs, as numpy doesn't know how to evaluate x * np.log(x)
            ind = np.where(populations > 0)[0]
            populations = populations[ind]
            log_prior_pops = log_prior_pops[ind]
            expr = populations.dot(np.log(populations)) - populations.dot(log_prior_pops)
            return -1 * regularization_strength * expr
        self.logp_prior = logp_prior

class DirichletBELT(BELT):
    """Bayesian Energy Landscape Tilting with Dirichlet prior."""
    def __init__(self, predictions, measurements, uncertainties, regularization_strength=1.0, prior_pops=None):
        """Bayesian Energy Landscape Tilting with Dirichlet prior.

        Parameters
        ----------
        predictions : ndarray, shape = (num_frames, num_measurements)
            predictions[j, i] gives the ith observabled predicted at frame j
        measurements : ndarray, shape = (num_measurements)
            measurements[i] gives the ith experimental measurement
        uncertainties : ndarray, shape = (num_measurements)
            uncertainties[i] gives the uncertainty of the ith experiment
        regularization_strength : float
            How strongly to weight the prior (e.g. lambda)
        precision : ndarray, optional, shape = (num_measurements, num_measurements)
            The precision matrix of the predicted observables.
        prior_pops : ndarray, optional, shape = (num_frames)
            Prior populations of each conformation.  If None, use uniform populations.
        """

        BELT.__init__(self,predictions,measurements,uncertainties,prior_pops=prior_pops)

        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))  # The prior on alpha is defined as a potential, so we use Uninformative variables here.
        self.initialize_variables()

        @pymc.potential
        def logp_prior(populations=self.populations):
            if populations.min() <= 0:
                return -1 * np.inf
            else:
                expr = self.prior_pops.dot(np.log(populations))
                return regularization_strength * expr
        self.logp_prior = logp_prior


def cross_validated_mcmc(predictions, measurements, uncertainties, model_factory, bootstrap_index_list, num_samples=50000, thin=1):
    """Fit model on training data, evaluate on test data, and return the chi squared.

    Parameters
    ----------
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    measurements : ndarray, shape = (num_measurements)
        measurements[i] gives the ith experimental measurement
    uncertainties : ndarray, shape = (num_measurements)
        uncertainties[i] gives the uncertainty of the ith experiment
    model_factory : lambda function
        A function that takes as input predictions, measurements, and uncertainties
        and generates a BELT model.  
    bootstrap_index_list : list (of integer numpy arrays)
        bootstrap_index_list is a list numpy arrays of frame indices that
        mark the different training and test sets.
        With a single trajectory, bootstrap_index_list will look something 
        like the following
        [np.array([0,1,2,... , n/2]), np.array([n / 2 + 1, ..., n - 1])]

    Returns
    -------
    train_chi, test_chi : float
        Training and test scores of cross validated models.

    """

    all_indices = np.concatenate(bootstrap_index_list)
    test_chi = []
    train_chi = []

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
