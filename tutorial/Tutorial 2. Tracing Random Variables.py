# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Tracing Additional Random Variables
# ===================================

# <markdowncell>

# Often, we would like to keep track of (i.e. `trace`) additional quantities while we sample the model likelihoods.  Tracing these quantities allows us to characterize the posterior distribution of arbitrary structural features.  For the case of tri-alanine, we would like to track the populations of each of the four conformational states.  To do so, we create a `pymc` random variable that monitors these populations.  By attaching this random variable as a member variable of our LVBP object, we can trace arbitrary random variables.  In this tutorial, we follow the same recipe as before, but add an extra random variable to output the populations of the $\alpha_r$, $\beta$, $PP_{II}$ $\;$, and $\alpha_l$ states.  

# <codecell>

import numpy as np
import pandas
import pymc 
import matplotlib.pyplot as plt
from fit_ensemble import lvbp, example_loader

measurements, predictions, uncertainties = example_loader.load_alanine()
phi, psi, assignments, indicators = example_loader.load_alanine_dihedrals()

num_samples = 25000  # Generate 25,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

regularization_strength = 3.0  # How strongly do we prefer a "uniform" ensemble (the "raw" MD)? 

lvbp_model = lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength)
lvbp_model.sample(num_samples, thin=thin, burn=burn)

# <markdowncell>

# Now we run the new code that outputs the trace of an additional "observable".  We simply call `lvbp_model.trace_observable()`.  This function takes a numpy array as input; here we input the transpose of the matrix of state indicators.  
# 
# Thus, for each conformational ensemble in our MCMC trace, we calculate the population of the four torsional states:

# <codecell>

state_pops_trace = lvbp_model.trace_observable(indicators.T)

# <markdowncell>

# We have calculated a trace of the state populations for each conformational ensemble in the MCMC chain.  We first characterize the average (over all MCMC samples) state populations.  We also look at the state populations of the raw MD simulation.

# <codecell>

state_pops_raw = np.bincount(assignments) / float(len(assignments))
state_pops = state_pops_trace.mean(0)

# <markdowncell>

# We would like to view the conformational populations as a table.  We use the pandas library to construct an object for a tabular view of the populations:

# <codecell>

pandas.DataFrame([state_pops_raw,state_pops],columns=[r"$PP_{II}$",r"$\beta$",r"$\alpha_r$",r"$\alpha_l$"],index=["Raw (MD)", "LVBP"])

# <markdowncell>

# It is also useful to look at the uncertainties associated with each of the state populations:

# <codecell>

state_uncertainties = state_pops_trace.std(0)
pandas.DataFrame([state_uncertainties],columns=[r"$PP_{II}$",r"$\beta$",r"$\alpha_r$",r"$\alpha_l$"],index=["LVBP"])

# <markdowncell>

# You've finished this tutorial.  In your own research, you can use a similar approach to characterize quantities like RMSD, radius of gyration, distances between atoms, side chain torsions, etc.  
# 
# Continue on to the third tutorial [here].  
# 
# References
# ----------
# 
# * Beauchamp, K. A., Das, R. , and Pande, V. S.  Inferring Conformational Ensembles from Noisy Experiments.  In Prep.
# 
# * Sosnick et al.  Helix, sheet, and polyproline II frequencies and strong nearest neighbor effects in a restricted coil library.  Biochemistry, 2005.

# <codecell>


