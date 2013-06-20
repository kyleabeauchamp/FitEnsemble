"""


"""
import abc
import numpy as np
import pymc
import utils

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
    block_pops[:-1] = prior_dirichlet.value[:]  # The pymc Dirichlet does not explicitly store the final component
    block_pops[-1] = 1.0 - block_pops.sum()  # Calculate the final component from normalization.

    prior_populations = np.ones(num_frames)
    for k,ind in enumerate(bootstrap_index_list):
        prior_populations[ind] = block_pops[k] / len(ind)
    
    return prior_populations        

class EnsembleFitter():
    """Abstract base class for ensemble modeling."""
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, predictions, measurements, uncertainties, prior_pops=None):
        """Abstract base class for ensemble modeling.

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
            
        Notes
        -----
        
        In subclasses of EnsembleFitter (e.g. BELT), the __init__() function 
        should take predictions, measurements, and uncertainties as arguments.
        Any additional inputs *must* have default values.  
        """
        utils.validate_input_arrays(predictions, measurements, uncertainties, prior_pops=prior_pops)
        self.uncertainties = uncertainties.astype('float64')  # Performance is improved by having everything the same dtype
        self.predictions = predictions.astype('float64')  # We could get 2X more performance with 32 bit, but not worth the precision issues
        self.measurements = measurements.astype('float64')
        
        self.num_frames, self.num_measurements = predictions.shape        
        self.prior_pops = get_prior_pops(self.num_frames, prior_pops)
                            
    def sample(self, num_samples, thin=1, burn=0, save_pops=False, filename = None):
        """Construct MCMC object and begin sampling."""        
        if filename == None:
            db = "ram"
        else:
            db = "hdf5"
            
        self.mcmc = pymc.MCMC(self, db=db, dbname=filename)
        self.save(db)
        self.mcmc.sample(num_samples, thin=thin, burn=burn)
        
    def save(self, db):
        """Save the input data to disk.
        
        Notes
        -----
        Saves predictions, measurements, observables, and prior_pops to the 
        HDF5 PyMC database.  
        """

        if db != "hdf5":
            return

        from tables import Float64Atom, Filters
        compression = Filters(complevel=9, complib='blosc', shuffle=True)
        F = self.mcmc.db._h5file

        F.createCArray("/", "predictions", Float64Atom(), self.predictions.shape, filters=compression)
        F.root.predictions[:] = self.predictions
        
        F.createCArray("/", "measurements", Float64Atom(), self.measurements.shape, filters=compression)
        F.root.measurements[:] = self.measurements

        F.createCArray("/", "uncertainties", Float64Atom(), self.uncertainties.shape, filters=compression)
        F.root.uncertainties[:] = self.uncertainties

        F.createCArray("/", "prior_pops", Float64Atom(), self.prior_pops.shape, filters=compression)
        F.root.prior_pops[:] = self.prior_pops

    def accumulate_populations(self):
        """Accumulate populations over MCMC trace.

        Returns
        -------
        p : ndarray, shape = (num_frames)
            Posterior averaged populations of each conformation
        """
        p = np.zeros(self.num_frames)        

        for pi in self.iterate_populations():
            p += pi
            
        p /= p.sum()

        return p

    def trace_observable(self, observable_features):
        """Calculate an function for each sample in the MCMC trace.

        Parameters
        ----------
        observable_features : ndarray, shape = (num_frames, num_features)
            observable_features[j, i] gives the ith feature of frame j

        Returns
        -------
        observable : ndarray, shape = (num_samples, num_features)
            The trace of the ensemble average observable for each MCMC sample.
        """
        observable = []

        for p in self.iterate_populations():
            observable.append(observable_features.T.dot(p))

        return np.array(observable)
        
    @classmethod
    def load(cls, filename):
        """Load a previous MCMC trace from a PyTables HDF5 file.
        
        Parameters
        ----------
        filename : string
            The filename for a previous run fit_ensemble MCMC trace.        
        """
        
        mcmc = pymc.database.hdf5.load(filename, 'r')

        F = mcmc._h5file
        predictions = F.root.predictions[:]
        measurements = F.root.measurements[:]
        uncertainties = F.root.uncertainties[:]
        prior_pops = F.root.prior_pops[:]        
        
        ensemble_fitter = cls(predictions, measurements, uncertainties, prior_pops=prior_pops)
        ensemble_fitter.mcmc = mcmc

        return ensemble_fitter
