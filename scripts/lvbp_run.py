#!/usr/bin/env python
import os
import argparse
import numpy as np
from fitensemble import lvbp
from fitensemble.ensemble_fitter import reduced_chi_squared

def load_numpy(filename):
    filetype = os.path.splitext(filename)[1]
    if filetype == ".dat":
        return np.loadtxt(filename)
    elif filetype == ".npz":
        return np.load(filename)["arr_0"]

def check_filetype(filename, acceptable_filetypes=[".dat", ".npz"]):
    filetype = os.path.splitext(filename)[1]
    if filetype not in acceptable_filetypes:
        raise argparse.ArgumentError("%s must be readable by Numpy and have filetype '.dat' or '.npz'" % filename)

#############################################
#Parsing input arguments
parser = argparse.ArgumentParser(description='Generate a conformation ensemble using LVBP')
parser.add_argument('-out', type=str,   help='Filename for MCMC trace (output)', metavar='str')
parser.add_argument('-pops', type=str,  help='Filename for maximum a posteriori populations (output)', metavar='str')
parser.add_argument('-expt', type=str,  help='Filename for for input experimental data.', metavar='str')
parser.add_argument('-sigma', type=str, help='Filename for for input uncertainties.', metavar='str')
parser.add_argument('-pred', type=str,  help='Filename for for input predicted experiments.', metavar='str')
parser.add_argument('-num', type=int,   help='Number of MCMC samples.', metavar='int')
parser.add_argument('-thin', type=int,  help='Subsample every nth sample in MCMC. Default 1', metavar='int',default=1)
parser.add_argument('-burn', type=int,  help='Discard first n MCMC samples. Default 0', metavar='int',default=0)
parser.add_argument('-reg', type=float, help='Weight of MaxEnt prior--the regularization strength.', metavar='float')


args = parser.parse_args()
check_filetype(args.expt)
check_filetype(args.sigma)
check_filetype(args.pred)

measurements = load_numpy(args.expt)
uncertainties = load_numpy(args.sigma)
predictions = load_numpy(args.pred)


S = lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, args.reg)

print("Running LVBP MCMC")
S.sample(args.num, thin=args.thin, burn=args.burn, filename=args.out)

print("Calculating maximum a posteriori populations.")
p = S.accumulate_populations()

print("Model summary")
print("Reduced chi squared of raw MD: %f" % reduced_chi_squared(predictions.mean(0), measurements, uncertainties))
print("Reduced chi squared of LVBP: %f" % reduced_chi_squared(predictions.T.dot(p), measurements, uncertainties))
print("Maximum population: %f at frame %d" % (p.max(), p.argmax()))
print("Average population: %f" % (p.mean()))
print("Minimum population: %f at frame %d" % (p.min(), p.argmin()))

np.savetxt(args.pops, p)
