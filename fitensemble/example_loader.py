import pandas as pd
import numpy as np
import pkg_resources

#subsample = 3  # By default, subsample ALA3 to make calculations faster for tutorial.

def assign_states(phi,psi):
    """
    Notes:
    State 0: PPII
    State 1: beta
    State 2: alpha
    State 3: gamma, alpha_l
        
    """
    ass = (0*phi).astype('int') + 3
    
    #States from Tobin Sosnick, Biochemistry.
    ass[(phi <= 0)&(phi>=-100)&((psi>=50.)|(psi<= -100))] = 0
    ass[(phi <= -100)&((psi>=50.)|(psi<= -100))] = 1
    ass[(phi <= 0)&((psi<=50.)&(psi>= -100))] = 2
    ass[(phi > 0)] = 3

    return ass

def J3_HN_HA(phi):
    """
    
    Notes:
        
        RMS = 0.36.  Karplus coefficients from Beat Vogeli, Jinfa Ying, Alexander Grishaev, and Ad Bax
    """
    phi = phi * np.pi/180.
    phi0 = -60 * np.pi/180.

    A = 8.4
    B = -1.36
    C = 0.33

    return A * np.cos(phi + phi0) ** 2. + B * np.cos(phi + phi0) + C

def load_alanine_pandas():
    """Load the predictions, measurements, and uncertainties.
    J coupling data from Baldwin, PNAS 2006.  
    """
    
    experiments_filename = pkg_resources.resource_filename("fitensemble","example_data/experiments.tab")
    
    experiments = pd.io.parsers.read_table(experiments_filename, sep="\s*", index_col=0)
    
    measurements = experiments["measurements"]
    uncertainties = experiments["uncertainties"]

    phi, psi, assignments, indicators = load_alanine_dihedrals()
    predictions = pd.DataFrame(J3_HN_HA(phi), columns=["J3_HN_HA"])

    return predictions, measurements, uncertainties

def load_alanine_numpy():
    """Load the predictions, measurements, and uncertainties.
    J coupling data from Baldwin, PNAS 2006.  
    """
    predictions, measurements, uncertainties = load_alanine_pandas()
    return predictions.values, measurements.values, uncertainties.values

def load_alanine_dihedrals():
    dih_filename = pkg_resources.resource_filename("fitensemble","example_data/rama.npz")
    phi, psi = np.load(dih_filename)["arr_0"]
    
    assignments = assign_states(phi,psi)
    indicators = np.array([assignments==i for i in xrange(4)])
    return phi, psi, assignments, indicators
