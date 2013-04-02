# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Cross-Validation
# ================
# 
# Here, we discuss using cross-validation to estimate the optimal value of `regularization_strength`.  In [1], we suggested three approaches to estimating `regularization_strength`:
# 
# * Cross-validating on the experimental data
# * Using $\chi^2$ analysis
# * Cross-validating on the simulation data
# 
# Because we have only a single experiment here, the first approach is not appropriate.  The second approach is also challenging because we have only approximate knowledge about the uncertainties, which we modelled as the RMS error found when fitting the Karplus equation.  We are thus left with the third approach.
# 
# To cross-validate using the simulation data, we must first divide the data into a list of training and test sets.  The key idea is that a model will be fit on the training data, but evaluated on the test set.  We choose `regularization_strength` to maximize the ($\chi^2$) performance on the test data.  In this way, we prevent overfitting and ensure that our model *generalizes* to arbitrary simulation datasets.  
# 
# To begin, we perform the same setup as previously:

# <codecell>

import numpy as np
import pandas
import matplotlib.pyplot as plt
from fit_ensemble import lvbp, example_loader

measurements, predictions, uncertainties = example_loader.load_alanine()
phi, psi, assignments, indicators = example_loader.load_alanine_dihedrals()

num_samples = 25000  # Generate 25,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

# <markdowncell>

# fit_ensemble provides a helper function (`cross_validated_mcmc`) to assist with cross validation.  However, the user must still provide a list of indices that divide the dataset into disjoint sets for cross-validation.  For a single MD trajectory, the simplest way to do so is by splitting the trajectory into its first and second halves.  The following code does just that:

# <codecell>

num_fold = 2
bootstrap_index_list = np.array_split(np.arange(len(predictions)), num_fold)

# <markdowncell>

# We're ready to do the cross-validation.  We will use a grid search, where we build models using multiple values of `regularization_strength`.  When the calculations have finished, the results will be displayed in tabular form.  

# <codecell>

regularization_strength_list = [1.0, 3.0, 5.0]
all_test_chi = np.zeros(3)
for k, regularization_strength in enumerate(regularization_strength_list):
    train_chi, test_chi = lvbp.cross_validated_mcmc(predictions, measurements, uncertainties, regularization_strength, bootstrap_index_list, num_samples=num_samples, thin=thin)
    all_test_chi[k] = test_chi.mean()
pandas.DataFrame([all_test_chi], columns=regularization_strength_list, index="$\chi^2$")

# <codecell>


# <markdowncell>

# In the above table, we see that the best fitting model is achieved when `regularization_strength` is 3.0.  This motivates the choice used in the previous two tutorials.  
# 
# Congratulations!  You've finished the third tutorial.  Continue on to the next tutorial here [X].  
# 
# References
# ----------
# 
# * Beauchamp, K. A., Das, R. , and Pande, V. S.  Inferring Conformational Ensembles from Noisy Experiments.  In Prep.

# <codecell>


