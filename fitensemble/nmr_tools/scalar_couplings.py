"""
This file contains scripts for calculating J Couplings from backbone dihedrals.

References:

Self-consistent 3J coupling analysis for the joint calibration of Karplus coefficients and evaluation of torsion angles

Limits on variations in protein backbone dynamics from precise measurements of scalar couplings

Structure and dynamics of the homologous series of alanine peptides: a joint molecular dynamics/NMR study
"""
import numpy as np
import pandas as pd


def J3_HN_HA_schwalbe(phi):
    """
    RMS = 0.39
    Personal RMS on ubiquitin = 0.254
    Originally from Hu and Bax
    """
    phi = phi*np.pi/180.
    phi0 = -60*np.pi/180.
    A = 7.09
    B = -1.42
    C = 1.55
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_HA_ruterjans(phi):
    """RMS = 0.25
    """
    phi = phi*np.pi/180.
    phi0 = -60*np.pi/180.
    A = 7.90
    B = -1.05
    C = 0.65
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_HA_bax(phi):
    """RMS = 0.36
    """
    phi = phi*np.pi/180.
    phi0 = -60*np.pi/180.
    A = 8.4
    B = -1.36
    C = 0.33
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_HA_karplus(phi):
    """RMS = 0.36
    """
    phi = phi*np.pi/180.
    phi0 = 0.0 * np.pi/180.
    A = 6.4
    B = -1.4
    C = 1.9
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_Cprime_schwalbe(phi):
    """
    RMS = 0.32
    Parms originally from Hu and Bax
    """
    phi = phi*np.pi/180.
    phi0 = 180*np.pi/180.
    phi1 = -60*np.pi/180.
    A = 4.29
    B = -1.01
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi1)


def J3_HN_Cprime_ruterjans(phi):
    """
    RMS = 0.39
    """
    phi = phi*np.pi/180.
    phi0 = 180*np.pi/180.
    A = 4.41
    B = -1.36
    C = 0.24
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_Cprime_bax(phi):
    """
    RMS = 0.30
    """
    phi = phi*np.pi/180.
    phi0 = 180*np.pi/180.
    A = 4.36
    B = -1.08
    C = -0.01
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HA_Cprime_schwalbe(phi):
    """
    RMS = 0.24
    Parms from Hu Bax
    """
    phi = phi*np.pi/180.
    phi0 = 120*np.pi/180.
    A = 3.72
    B = -2.18
    C = 1.28
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HA_Cprime_ruterjans(phi):
    """
    RMSD = 0.44
    """
    phi = phi*np.pi/180.
    phi0 = 120*np.pi/180.
    A = 3.76
    B = -1.63
    C = 0.89
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_Cprime_Cprime_schwalbe(phi):
    """
    RMS = 0.13

    Parms from Hu and Bax
    """
    phi = phi*np.pi/180.
    phi0 = 0.0*np.pi/180.
    A = 1.36
    B = -0.93
    C = 0.60
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_Cprime_Cprime_ruterjans(phi):
    """
    RMS = 0.30
    """
    phi = phi*np.pi/180.
    phi0 = 0.0*np.pi/180.
    A = 1.51
    B = -1.09
    C = 0.52
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_CB_schwalbe(phi):
    """
    RMS = 0.21

    Parms from Hu Bax
    """
    phi = phi*np.pi/180.
    phi0 = 60*np.pi/180.
    A = 3.06
    B = -0.74
    C = 0.13
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_CB_ruterjans(phi):
    """
    RMS = 0.25
    """
    phi = phi*np.pi/180.
    phi0 = 60*np.pi/180.
    A = 2.90
    B = -0.56
    C = 0.18
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J3_HN_CB_bax(phi):
    """
    RMS = 0.22
    """
    phi = phi*np.pi/180.
    phi0 = 60*np.pi/180.
    A = 3.71
    B = -0.59
    C = 0.08
    return A*np.cos(phi + phi0)**2. + B*np.cos(phi + phi0) + C


def J1_N_CA_schwalbe(psi):
    """
    RMS = 0.52659745254609414
    RMS estimated from extracting data from figure 4A
    Parms originally from Wirmer and Schwalbe
    """
    psi = psi*np.pi/180.
    psi0 = 0.0*np.pi/180.
    A = 1.70
    B = -0.98
    C = 9.51
    return A*np.cos(psi + psi0)**2. + B*np.cos(psi + psi0) + C


def J2_N_CA_schwalbe(psi):
    """
    RMS = 0.4776
    Parms originally from Ding and Gronenborn
    """
    psi = psi*np.pi/180.
    psi0 = 0.0*np.pi/180.
    A = -0.66
    B = -1.52
    C = 7.85
    return A*np.cos(psi + psi0)**2. + B*np.cos(psi + psi0) + C


def J3_HN_CA_ruterjans(phi,psi):
    phi = phi*np.pi/180.
    psi = psi*np.pi/180.
    c = np.cos
    s = np.sin
    f = -0.23*c(phi) - 0.2*c(psi) + 0.07*s(phi) + 0.08*s(psi) + 0.07*c(phi)*c(psi) + 0.12*c(phi)*s(psi) - 0.08*s(phi)*c(psi) - 0.14*s(phi)*s(psi) + 0.54
    return f


def J3_HA_HA_CYS(chi):
    t = chi * np.pi / 180.
    c = np.cos
    f = 5.32 - 1.37 * c(t) + 3.61*c(2*t)
    return f

J3_HN_HA = J3_HN_HA_bax
J3_HN_Cprime = J3_HN_Cprime_bax
J3_HN_CB = J3_HN_CB_bax
J1_N_CA = J1_N_CA_schwalbe
J2_N_CA = J2_N_CA_schwalbe
J3_Cprime_Cprime = J3_Cprime_Cprime_ruterjans
J3_HA_Cprime = J3_HA_Cprime_ruterjans

uncertainties = pd.Series({
"J3_HN_HA":0.36, "J3_HN_Cprime":0.30, "J3_HN_CB":0.22, "J1_N_CA":0.53, "J2_N_CA":0.48,"J3_HA_Cprime":0.44
})
