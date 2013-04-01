# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# File Input and Output
# =====================
# 
# In this tutorial, we discuss saving an LVBP model to disk for later use.  We repeat the same calculations as in the previous tutorials, but this time save the model to disk for later use.  
# 

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from fit_ensemble import lvbp, example_loader

measurements, predictions, uncertainties = example_loader.load_alanine()
phi, psi, assignments, indicators = example_loader.load_alanine_dihedrals()

num_samples = 25000  # Generate 25,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

regularization_strength = 3.0  # How strongly do we prefer a "uniform" ensemble (the "raw" MD)? 

# <markdowncell>

# The following code builds a model, just like before.  However, we now pass a filename argument when we begin the MCMC sampling.  This tells PyMC to save the results to disk as an HDF5 database.  This is useful for situations where the entire MCMC trace cannot fit in system memory.  

# <codecell>

lvbp_model = lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength)

pymc_filename = "trialanine.h5"
lvbp_model.sample(num_samples, thin=thin, burn=burn,filename=pymc_filename)

# <markdowncell>

# The HDF5 database file trialanine.h5 contains all components necessary to work with your LVBP model.  This includes
# 
# * `predictions`
# * `measurements`
# * `uncertainties`
# * The MCMC trace
# 
# To load an HDF5 file from disk, use the load() function:

# <codecell>

lvbp_model = lvbp.MaxEnt_LVBP.load("./trialanine.h5")

# <markdowncell>

# As we did previously, this model can be used to calculate the MAP conformational populations (`p`) or a trace (`state_pops_trace`) of an arbitrary observable:

# <codecell>

p = lvbp_model.accumulate_populations()
state_pops_trace = lvbp_model.trace_observable(indicators.T)

