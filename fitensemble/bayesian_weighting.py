"""
An implementation of the Bayesian weighting approach outlined in Stultz et al,
JACS. 2010.

Note: we have chosen to use a Dirichlet prior for the population vector.
Our implementation may have other slight design changes.  We are *not*
claiming to exactly reproduce the BW method as published.
"""
import numpy as np
import pymc
from ensemble_fitter import EnsembleFitter


def framewise_to_statewise(predictions, state_assignments):
    num_states = state_assignments.max() + 1
    predictions_statewise = np.array([predictions[state_assignments == i].mean(0) for i in np.arange(num_states)])
    return predictions_statewise


def get_chi2(populations, predictions, measurements, uncertainties, mu=None):
    """Return the chi squared objective function.

    Parameters
    ----------
    alpha : ndarray, shape = (num_measurements)
        Biasing weights for each experiment.
    predictions : ndarray, shape = (num_states, num_measurements)
        predictions[j, i] gives the ith observabled predicted at state j
    prior_pops : ndarray, shape = (num_states)
        Prior populations of each conformation.  If None, then use uniform pops.

    Returns
    -------
    populations : ndarray, shape = (num_states)
        Reweighted populations of each conformation

    """

    if mu == None:
        mu = predictions.T.dot(populations)

    delta = (measurements - mu) / uncertainties

    return np.linalg.norm(delta)**2.


class BayesianWeighting(EnsembleFitter):
    def __init__(self, predictions, measurements, uncertainties, state_assignments, prior_pops=None):
        """Note that this class state-averaged predictions.  This is in contrast to the BELT methed, which
        takes inputs for each *frame*.
        """
        EnsembleFitter.__init__(self, predictions, measurements, uncertainties)

        self.state_assignments = state_assignments
        self.num_states = len(np.unique(self.state_assignments))

        if prior_pops is not None:
            self.prior_pops = prior_pops
        else:
            self.prior_pops = np.ones(self.num_states)

        self.initialize_variables()

    def initialize_variables(self):
        """Initializes MCMC variables."""
        self.dirichlet = pymc.Dirichlet("dirichlet", self.prior_pops)  # This has size (n-1), so it is missing the final component.
        self.matrix_populations = pymc.CompletedDirichlet("matrix_populations", self.dirichlet)  # This RV fills in the missing value of the population vector, but has shape (1, n) rather than (n)
        self.populations = pymc.CommonDeterministics.Index("populations", self.matrix_populations, 0)  # Finally, we get a flat array of the populations.

        self.dirichlet.keep_trace = False

        @pymc.dtrm
        def mu(populations=self.populations):
            return populations.dot(self.predictions)
        self.mu = mu

        @pymc.potential
        def logp(populations=self.populations,mu=self.mu):
            return -1 * get_chi2(populations, self.predictions, self.measurements, self.uncertainties, mu=mu)
        self.logp = logp

    def iterate_populations(self):
        for i, pi in enumerate(self.mcmc.trace("matrix_populations")):
            yield pi[0]  # We have some shape issues from using Dirichlets.

    def sample(self, num_samples, thin=1, burn=0):
        """Construct MCMC object and begin sampling.

        Notes
        -----
        BW presently does not support saving to a file.
        """
        self.mcmc = pymc.MCMC(self)
        self.mcmc.sample(num_samples, thin=thin, burn=burn)
