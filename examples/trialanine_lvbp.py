import numpy as np
from fit_ensemble import lvbp, example_loader
import pymc
import matplotlib.pyplot as plt

pymc_filename = "./trialanine_MCMC.h5"

regularization_strength = 5.0  # How strongly do we prefer a "uniform" ensemble (the "raw" MD)?
num_samples = 25000  # Generate 100,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

# Load the measured and predicted observables. Here, we use a single J coupling.
keys, measurements, predictions, uncertainties = example_loader.load_alanine(subsample=5)  # Subsample the data 5X to make the calculation faster.

# Load the backbone torsions and state assignments
phi, psi, assignments, indicators = example_loader.load_alanine_dihedrals(subsample=5)  # Subsample the data 5X to make the calculation faster.

#Create an LVBP object
S = lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength)

#  Here, we add a "tracker" random variable that computes the populations of each conformational state (PPII, beta, alpha, other)
S.state_pops = pymc.Lambda("state_pops",lambda populations=S.populations: np.array([populations[indicators[i]].sum() for i in xrange(4)]))

#  Use MCMC to sample the likelihood.
S.sample(num_samples, thin=thin, burn=burn, filename=pymc_filename)

#  Calculate the maximum a posteriori conformational populations.
p = S.accumulate_populations()

print("Reduced chi squared of raw MD: %f" % np.mean(((predictions.mean(0) - measurements) / uncertainties)**2))
print("Reduced chi squared of LVBP: %f" % np.mean(((predictions.T.dot(p) - measurements) / uncertainties)**2))


#  Check the MCMC trace of alpha for convergence

a0 = S.mcmc.trace("alpha")[:,0]
plt.plot(a0)
plt.title("MCMC trace of alpha")


#  Look at the conformational properties of the raw and LVBP models

h,x,y = np.histogram2d(phi,psi, bins=100)
extent = [-180,180,-180,180]
plt.figure()
plt.xlabel(r"""$\chi_1 CYS14$ [$\circ$]""")
plt.ylabel(r"""$\chi_1 CYS38$ [$\circ$]""")
plt.title(r"MD Population Density")
plt.imshow(h.T,origin='lower',extent=extent)
plt.xlim(-180,180)
plt.ylim(-180,180)



h,x,y = np.histogram2d(phi,psi,bins=100,weights=p)
plt.figure()
plt.xlabel(r"""$\chi_1 CYS14$ [$\circ$]""")
plt.ylabel(r"""$\chi_1 CYS38$ [$\circ$]""")
plt.title(r"Reweighted Population Density")
plt.imshow(h.T,origin='lower',extent=extent)
plt.xlim(-180,180)
plt.ylim(-180,180)


mu = S.mcmc.trace("state_pops")[:].mean(0)
sigma = S.mcmc.trace("state_pops")[:].std(0)
mu0 = np.bincount(assignments) / float(len(assignments))

