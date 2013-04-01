# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Modelling Conformational Ensembles with fit_ensemble
# ==================================================
# 
# Introduction
# ------------
# 
# In this tutorial, we use Linear Virtual Biasing Potential (LVBP) to infer the conformational ensemble of tri-alanine.  This calculation is similar to the one done in [1], but uses only a single experiment (for speed reasons).  
# 
# Performing an LVBP calculation requires three inputs:
# 
# * A set of conformations to be used as a "starting" guess for the ensemble: $x_i$
# 
# * A set of experimental ``measurements`` and their ``uncertainties``
# 
# * The predicted experimental observables (``predictions``) at each conformation.
# 
# LVBP combines `measurements`, `uncertainties`, and `predictions` using Bayesian inference to produce a conformational ensemble that best agrees with your data.  
# 
# 
# Importing fit_ensemble
# ----------------------
# 
# We first import some Python libraries used in the tutorial

# <codecell>

import numpy as np
import pymc 
import matplotlib.pyplot as plt
from fit_ensemble import lvbp, example_loader

# <markdowncell>

# Load input data
# ---------------
# 
# We will use a single $^3J(H^N H^\alpha )$ $ \;$  NMR measurement to reweight molecular dynamics simulations performed in the Amber ff96 forcefield.  Note that a single experiment is *not* sufficient to correct the forcefield bias in ff96.  However, the single-experiment calculation illustrates the key points and runs in under a minute.  
# 
# We next load the data used to connect simulation and experiment.  The tri-alanine data is loaded by a helper function (`load_alanine`) included in fit_ensemble for the purpose of simplifying the tutorial.  

# <codecell>

measurements, predictions, uncertainties = example_loader.load_alanine()

# <markdowncell>

# In addition to the quantities *necessary* for the LVBP calculation, it's also useful to examine additional structural properties such as the $\phi$ and $\psi$ backbone torsions.  The follow code loads the backbone torsion angles $\phi$ and $\psi$ for the internal alanine in tri-alanine.  In addition, we have also calculated the conformation states of each snapshot ($\alpha_r$, $\beta$, $PP_{II}$ $\;$, or $\alpha_l$), using the state definitions from [2].  

# <codecell>

phi, psi, assignments, indicators = example_loader.load_alanine_dihedrals()

# <markdowncell>

# We also need to define a number of parameters for the LVBP calculation.  These include the number of MCMC samples to generate, the amount by which to thin (i.e. subsample) the traces, and the number of samples to discard during the equilibration:

# <codecell>

num_samples = 25000  # Generate 25,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

# <markdowncell>

# We also need to define the amount of regularization to use.  We will use maximum entropy regularization, which prefers ensembles that weight each conformation uniformly. With a large value of `regularization_strength` ($\lambda \;$), we strongly prefer the raw MD ensemble, which gives equal population to each conformation.

# <codecell>

regularization_strength = 3.0  

# <markdowncell>

# At this point, we are almost ready to go.  We create an LVBP object using our simulation and experimental data:

# <codecell>

lvbp_model = lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength)

# <markdowncell>

# We are good to go!  We now sample the likelihood of models given our data:

# <codecell>

lvbp_model.sample(num_samples, thin=thin, burn=burn)

# <markdowncell>

# Any calculation performed using Markov chain Monte Carlo should be checked for convergence.  There are many sophisticated tests of convergence, but a powerful heuristic is to graphically examine the trace of model parameters $\alpha$.  A properly sampled and thinned MCMC trace should look like white noise:

# <codecell>

a0 = lvbp_model.mcmc.trace("alpha")[:,0]
plt.plot(a0)
plt.title("MCMC trace of alpha")

# <markdowncell>

# The key outputs of an LVBP calculation are the maximum a posteriori populations.  These are essentially the *most likely* conformational populations for each conformation in your prior ensemble.  The `accumulate_populations()` member function calculates these populations for you:

# <codecell>

p = lvbp_model.accumulate_populations()

# <markdowncell>

# You should now check the agreement between the simulation and experiment.  We check both the raw MD simulation and the LVBP model.  

# <codecell>

print("Reduced chi squared of raw MD: %f" % np.mean(((predictions.mean(0) - measurements) / uncertainties)**2))
print("Reduced chi squared of LVBP: %f" % np.mean(((predictions.T.dot(p) - measurements) / uncertainties)**2))

# <markdowncell>

# The above verifies that our LVBP model gives excellent agreement with the available experimental data--much more so than the raw MD.  At this point, we can dig deeper into understanding the results of our calculation.  Because we are looking at a single residue, we can visualize the relevant populations using a Ramachandran plot of the $\phi$ and $\psi$ torsion angles.  We first look at the raw MD simulation.

# <codecell>

h,x,y = np.histogram2d(phi,psi, bins=100)
extent = [-180,180,-180,180]
plt.figure()
plt.xlabel("$\phi$")
plt.ylabel("$\psi$")
plt.title(r"MD Population Density")
plt.imshow(h.T,origin='lower',extent=extent)
plt.xlim(-180,180);
plt.ylim(-180,180);

# <markdowncell>

# We now look at the reweighted (i.e. LVBP) ensemble.  You will notice that the $PP_{II}$ $\;$ basin (top center-right) has increased its population, while the $\beta$ basin (top left) has decreased its population.  We therefore conclude that the Amber ff96 forcefield is overly $\beta$-prone; the LVBP model has partially corrected this bias.  

# <codecell>

h,x,y = np.histogram2d(phi,psi,bins=100,weights=p)
plt.figure()
plt.xlabel("$\phi$")
plt.ylabel("$\psi$")
plt.title(r"Reweighted Population Density")
plt.imshow(h.T,origin='lower',extent=extent)
plt.xlim(-180,180);
plt.ylim(-180,180);

# <markdowncell>

# Congratulations!  You've finished the first tutorial.  Continue on to the next tutorial here [X].  
# 
# References
# ----------
# 
# * Beauchamp, K. A., Das, R. , and Pande, V. S.  Inferring Conformational Ensembles from Noisy Experiments.  In Prep.

