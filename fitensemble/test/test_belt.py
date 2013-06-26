from fitensemble import belt, ensemble_fitter
from mdtraj.testing import eq
from unittest import skipIf
import os
import numpy as np


def generate_gaussian_data(num_frames, num_measurements):
    prior_pops = ensemble_fitter.get_prior_pops(num_frames)
    predictions = np.random.normal(size=(num_frames, num_measurements))
    return prior_pops, predictions

def generate_uniform_data(num_frames, num_measurements):
    prior_pops = ensemble_fitter.get_prior_pops(num_frames)
    predictions = np.random.uniform(0.0, 1.0, size=(num_frames, num_measurements))
    return prior_pops, predictions

def generate_exponential_data(num_frames, num_measurements, scale):
    prior_pops = ensemble_fitter.get_prior_pops(num_frames)
    predictions = np.random.exponential(scale, size=(num_frames, num_measurements))
    return prior_pops, predictions

def test_get_populations_from_alpha_1D_gaussian():
    prior_pops, predictions = generate_gaussian_data(5000000, 1)
    alpha = np.array([0.25])
    populations = belt.get_populations_from_alpha(alpha, predictions, prior_pops)
    mu = predictions.T.dot(populations)
    eq(mu, -1.0 * alpha, decimal=3)
    
def test_get_populations_from_alpha_1D_uniform():
    prior_pops, predictions = generate_uniform_data(5000000, 1)
    alpha = np.array([0.25])
    populations = belt.get_populations_from_alpha(alpha, predictions, prior_pops)
    mu = predictions.T.dot(populations)
    a = alpha
    mu0 = (1. / a) - 1. / (np.exp(a) - 1.)
    eq(mu0, mu, decimal=3)

def test_get_populations_from_alpha_1D_exponential():
    scale = 3.0
    prior_pops, predictions = generate_exponential_data(5000000, 1, scale)
    alpha = np.array([0.25])
    populations = belt.get_populations_from_alpha(alpha, predictions, prior_pops)
    mu = predictions.T.dot(populations)
    mu0 = (alpha + scale ** -1.)**-1.
    eq(mu0, mu, decimal=3)

def test_get_populations_from_alpha_2D_gaussian():
    prior_pops, predictions = generate_gaussian_data(5000000, 2)
    alpha = np.array([-0.25, 0.25])
    populations = belt.get_populations_from_alpha(alpha, predictions, prior_pops)
    mu = predictions.T.dot(populations)
    eq(mu, -1.0 * alpha, decimal=3)

@skipIf(os.environ.get("TRAVIS", None) == 'true', "This SSE3 C code doesn't run correctly on travis-ci.org?") 
def test_BELT_1D_gaussian_maxent():
    num_frames = 400000
    predictions = np.random.normal(size=(num_frames,1))

    reg = 0.5
    measurements = np.array([0.25])
    uncertainties = np.array([1.0])

    model = belt.MaxEnt_BELT(predictions, measurements, uncertainties, regularization_strength=reg)
    model.sample(100000, thin=5)

    a = model.mcmc.trace("alpha")[:]
    mu = a.mean(0)
    sig = a.std(0)

    rho = np.array([(1 + reg) ** -0.5])
    mu0 = - measurements * rho ** 2.
    
    eq(mu, mu0, decimal=2)
    eq(sig, rho, decimal=2)

@skipIf(os.environ.get("TRAVIS", None) == 'true', "This SSE3 C code doesn't run correctly on travis-ci.org?") 
def test_BELT_1D_gaussian_MVN():
    num_frames = 400000
    predictions = np.random.normal(size=(num_frames,1))

    reg = 0.5
    measurements = np.array([0.25])
    uncertainties = np.array([1.0])

    model = belt.MVN_BELT(predictions, measurements, uncertainties, regularization_strength=reg)
    model.sample(100000, thin=5)

    a = model.mcmc.trace("alpha")[:]
    mu = a.mean(0)
    sig = a.std(0)

    rho = np.array([(1 + reg) ** -0.5])
    mu0 = - measurements * rho ** 2.
    
    eq(mu, mu0, decimal=2)
    eq(sig, rho, decimal=2)
