import numpy as np
from mdtraj.utils.arrays import ensure_type

def validate_pandas_columns(predictions, measurements, uncertainties):
    """Check that pandas DataFrames / Series have compatible entries.

    Parameters
    ----------
    predictions : pd.DataFrame, shape = (num_frames, num_measurements)
        predictions gives the ith observabled predicted at frame j
    measurements : pd.Series, shape = (num_measurements)
        The n experimental measurements
    uncertainties : pd.Series, shape = (num_measurements)
        The n experimental uncertainties

    Notes
    -----
    This function raises a TypeError if the columns of predictions are not
    identical to the index of measurements and uncertainties.
    """
    if not (predictions.columns.equals(measurements.index) and predictions.columns.equals(uncertainties.index)):
        raise(TypeError("predictions.columns, measurements.index, and uncertainties.index must be identical.  Perhaps your data is mis-aligned?"))

def validate_input_arrays(predictions, measurements, uncertainties, prior_pops=None):
    """Check input data for correct shape and dtype

    Parameters
    ----------
    predictions : ndarray, shape = (num_frames, num_measurements)
        predictions[j, i] gives the ith observabled predicted at frame j
    measurements : ndarray, shape = (num_measurements)
        measurements[i] gives the ith experimental measurement
    uncertainties : ndarray, shape = (num_measurements)
        uncertainties[i] gives the uncertainty of the ith experiment
    prior_pops : ndarray, shape = (num_frames), optional
        Prior populations of each conformation.  If None, skip.
    
    Notes
    -----
    All inputs must have float64 type and compatible shapes.
    """
    num_frames, num_measurements = predictions.shape

    ensure_type(predictions, np.float64, 2, "predictions")
    ensure_type(measurements, np.float64, 1, "measurements", shape=(num_measurements,))
    ensure_type(uncertainties, np.float64, 1, "uncertainties", shape=(num_measurements,))

    if prior_pops is not None:
        ensure_type(prior_pops, np.float64, 1, "prior_pops", shape=(num_frames,))
