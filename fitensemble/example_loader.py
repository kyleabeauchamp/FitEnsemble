import numpy as np
import pkg_resources

subsample = 3  # By default, subsample ALA3 to make calculations faster for tutorial.

def assign_states(phi,psi):
    """
    Notes:
    State 0: PPII
    State 1: beta
    State 2: alpha
    State 3: gamma, alpha_l
        
    """
    ass = (0*phi).astype('int') + 3
    
    #States from Tobin
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

def load_alanine(subsample=subsample):
    measurements = np.array([5.68])  # J coupling from Baldwin, PNAS 2006.  Table 1. Carbon CS from Joanna Long, 2004.

    dih_filename = pkg_resources.resource_filename("fitensemble","example_data/rama.npz")
    phi, psi = np.load(dih_filename)["arr_0"]
    J = J3_HN_HA(phi)

    uncertainties = np.array([0.36])
    
    predictions = np.array([J]).T
    
    return measurements, predictions[::subsample], uncertainties

def load_alanine_dihedrals(subsample=subsample):
    dih_filename = pkg_resources.resource_filename("fitensemble","example_data/rama.npz")
    phi, psi = np.load(dih_filename)["arr_0"]
    
    assignments = assign_states(phi,psi)
    indicators = np.array([assignments==i for i in xrange(4)])
    return phi[::subsample], psi[::subsample], assignments[::subsample], indicators[:, ::subsample]
