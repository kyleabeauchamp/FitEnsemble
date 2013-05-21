"""
An implementation of the Bayesian weighting approach outlined in Stultz et al,
JACS. 2010.  

Note: we have chosen to use a Dirichlet prior for the population vector.  
"""
import numpy as np
import pymc
from ensemble_fitter import EnsembleFitter
    
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


class BayesianWeighting(EnsembleFitter):
    def __init__(self, predictions, measurements, uncertainties, state_assignments, prior_state_pops=None):
        EnsembleFitter.__init__(self, predictions, measurements, uncertainties)
        
        self.state_assignments = state_assignments
        self.num_states = len(np.unique(self.state_assignments))

        if prior_state_pops is not None:
            self.prior_state_pops = prior_state_pops
        else:
            #self.prior_state_pops = np.bincount(state_assignments).astype('float')
            #self.prior_state_pops /= self.prior_state_pops.sum()
            self.prior_state_pops = np.ones(self.num_states)

        self.initialize_variables()

    def initialize_variables(self):
        """Initializes MCMC variables."""        
        self.dirichlet = pymc.Dirichlet("dirichlet", self.prior_state_pops)  # This has size (n-1), so it is missing the final component.  
        self.matrix_populations = pymc.CompletedDirichlet("matrix_populations", self.dirichlet)  # This RV fills in the missing value of the population vector, but has shape (1, n) rather than (n)
        self.populations = pymc.CommonDeterministics.Index("populations", self.matrix_populations, 0)  # Finally, we get a flat array of the populations.
        
        self.predictions_statewise = np.array([self.predictions[self.state_assignments == i].mean(0) for i in np.arange(self.num_states)])

        self.dirichlet.keep_trace = False
        #self.matrix_populations.keep_trace = False
        
        @pymc.dtrm
        def mu(populations=self.populations):
            return populations.dot(self.predictions_statewise)
        self.mu = mu

        @pymc.potential
        def logp(populations=self.populations,mu=self.mu):
            return -1 * get_chi2(populations, self.predictions, self.measurements, self.uncertainties, mu=mu)
        self.logp = logp
        
    def iterate_populations(self):
        for i, pi in enumerate(self.mcmc.trace("matrix_populations")):
            yield pi[0]  # We have some shape issues from using Dirichlets.

    def accumulate_populations(self):
        """Accumulate populations over MCMC trace.

        Returns
        -------
        p : ndarray, shape = (num_states)
            Posterior averaged populations of each conformation
        """
        p = np.zeros(self.num_states)        

        for pi in self.iterate_populations():
            p += pi
            
        p /= p.sum()

        return p
