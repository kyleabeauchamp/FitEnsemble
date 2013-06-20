from fitensemble import belt
from mdtraj.testing import eq
import numpy as np

def generate_gaussian_data(num_frames, num_measurements):
    prior_pops = belt.get_prior_pops(num_frames)
    predictions = np.random.normal(size=(num_frames, num_measurements))
    return prior_pops, predictions

def test_get_populations_from_alpha_1D_gaussian():
    prior_pops, predictions = generate_gaussian_data(5000000, 1)
    alpha = np.array([0.25])
    populations = belt.get_populations_from_alpha(alpha, predictions, prior_pops)
    mu = predictions.T.dot(populations)
    eq(mu, -1.0 * alpha, decimal=3)

def test_get_populations_from_alpha_2D_gaussian():
    prior_pops, predictions = generate_gaussian_data(5000000, 2)
    alpha = np.array([-0.25, 0.25])
    populations = belt.get_populations_from_alpha(alpha, predictions, prior_pops)
    mu = predictions.T.dot(populations)
    eq(mu, -1.0 * alpha, decimal=3)
