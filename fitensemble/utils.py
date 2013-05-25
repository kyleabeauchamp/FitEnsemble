import pandas as pd
import numpy as np
from mdtraj.utils.arrays import ensure_type

def validate_pandas_columns(predictions, measurements, uncertainties):
    if not (predictions.columns.equals(measurements.index) and predictions.columns.equals(uncertainties.index)):
        raise(TypeError("predictions.columns, measurements.index, and uncertainties.index must be identical.  Perhaps your data is mis-aligned?"))

def validate_input_arrays(predictions, measurements, uncertainties, prior_pops=None):
    num_frames, num_measurements = predictions.shape

    ensure_type(predictions, np.float64, 2, "predictions")
    ensure_type(measurements, np.float64, 1, "measurements", shape=(num_measurements,))
    ensure_type(uncertainties, np.float64, 1, "uncertainties", shape=(num_measurements,))

    if prior_pops is not None:
        ensure_type(prior_pops, np.float64, 1, "prior_pops", shape=(num_frames,))
