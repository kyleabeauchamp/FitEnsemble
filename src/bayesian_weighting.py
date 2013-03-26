"""
An implementation of the Bayesian weighting approach outlined in Stultz et al,
JACS. 2010.  

Note: we have chosen to use a Dirichlet prior for the population vector.  
"""
import numpy as np
import pymc
from ensemble import Ensemble


    
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

class BayesianWeighting(Ensemble):
    def __init__(self, predictions, measurements, uncertainties, prior_pops=None):
        Ensemble.__init__(self, predictions, measurements, uncertainties, prior_pops=prior_pops)

    def initialize_variables(self):
        """Initializes MCMC variables."""
        self.populations = pymc.Dirichlet("prior_dirichlet", np.ones(self.num_frames))
                
        @pymc.dtrm
        def mu(populations=self.populations):
            return populations.dot(self.predictions)
        self.mu = mu

        @pymc.potential
        def logp(populations=self.populations,mu=self.mu):
            return -1 * get_chi2(populations, self.predictions, self.measurements, self.uncertainties,mu=mu)
        self.logp = logp