import numpy as np
import fit_ensemble

f_sim = np.load("./f_sim.npz")["arr_0"]
f_exp = np.load("./f_exp.npz")["arr_0"]

alpha = np.zeros(len(f_exp))
prior_sigma = 0.7

alpha = fit_ensemble.minimize_chi2(alpha,f_sim,f_exp,prior_sigma)