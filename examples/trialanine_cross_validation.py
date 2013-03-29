import numpy as np
from fit_ensemble import lvbp, example_loader
import matplotlib.pyplot as plt

num_samples = 100000  # Generate 100,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

# Load the measured and predicted observables. Here, we use a single J coupling.
keys, measurements, predictions, uncertainties = example_loader.load_alanine(subsample=5)  # Subsample the data 5X to make the calculation faster.

all_test_chi = []
all_train_chi = []

#  We build models at the following five regularization strengths
regularization_strength_list = [1.0, 5.0, 7.0, 9.0, 12.0]

for regularization_strength in regularization_strength_list:
    bootstrap_index_list = np.array_split(np.arange(len(predictions)), 2) # Create two datasets consisting of the first and second halves of the trajectory.
    train_chi, test_chi = lvbp.cross_validated_mcmc(predictions, measurements, uncertainties, regularization_strength, bootstrap_index_list, num_samples)    
    all_test_chi.append(test_chi.mean())
    all_train_chi.append(train_chi.mean())


plt.plot(regularization_strength_list, all_test_chi,'o')
plt.xlabel("Regularization Strength")
plt.ylabel("RMS Error")

#  The plot suggests we select 5.0 to achieve the highest accuracy model.