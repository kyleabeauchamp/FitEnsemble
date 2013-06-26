"""
Note: the tools in this file are UNTESTED and may not work.  There could be
bugs, or the theory could even be wrong.  Double check before using in
production environments.
"""

import numpy as np
import scipy.sparse
import pymc
from fitensemble.belt import BELT

class Jeffreys_BELT(BELT):
    """Bayesian Energy Landscape Tilting with Jeffrey's prior."""
    def __init__(self, predictions, measurements, uncertainties, prior_pops=None, weights_alpha=None):
        """Bayesian Energy Landscape Tilting with Jeffrey's prior.

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
            
        Notes:
        ------
        This feature is UNTESTED.            
        """
        BELT.__init__(self, predictions, measurements, uncertainties, prior_pops=prior_pops)

        self.alpha = pymc.Uninformative("alpha",value=np.zeros(self.num_measurements))
        self.initialize_variables()

        @pymc.potential
        def logp_prior(populations=self.populations,mu=self.mu):
            return log_jeffreys(populations,predictions,mu=mu)
        self.logp_prior = logp_prior


class MaxEnt_Correlation_Corrected_BELT(BELT):
    """Bayesian Energy Landscape Tilting with maximum entropy prior and correlation-corrected likelihood."""
    def __init__(self, predictions, measurements, uncertainties, regularization_strength=1.0, precision=None, prior_pops=None):
        """Bayesian Energy Landscape Tilting with maximum entropy prior and correlation-corrected likelihood.

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
            return -0.5 * chi2
        self.logp = logp


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

    Notes:
    ------
    This feature is UNTESTED.        
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
        
    Notes:
    ------
    This feature is UNTESTED.
    """

    I = fisher_information(populations,predictions,mu)
    sign,logdet = np.linalg.slogdet(I)
    return 0.5 * logdet
