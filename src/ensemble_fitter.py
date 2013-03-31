"""


"""
import abc
import numpy as np
import pymc

def reduced_chi_squared(predictions, measurements, uncertainties):
    return np.mean(((predictions.mean(0) - measurements) / uncertainties)**2)

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

class EnsembleFitter():
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
        self.prior_pops = get_prior_pops(self.num_frames, prior_pops)
                            
    def sample(self, num_samples, thin=1, burn=0,save_pops=False,filename = None):
        """Construct MCMC object and begin sampling."""
        if save_pops == False:
            self.populations.keep_trace = False
        
        if filename == None:
            db = "ram"
        else:
            db = "hdf5"
            
        self.mcmc = pymc.MCMC(self, db=db, dbname=filename)
        self.mcmc.sample(num_samples, thin=thin, burn=burn)
        
    def load(self, filename):
        """Load a previous MCMC trace from a PyTables HDF5 file.
        
        Parameters
        ----------
        filename : string
            The filename for a previous run fit_ensemble MCMC trace.        
        """
        self.mcmc = pymc.database.hdf5.load(filename)
