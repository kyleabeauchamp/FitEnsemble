.. _command_line:

############################
Running via the Command Line 
############################

:Release: |version|
:Date: |today|


Running FitEnsemble via the command line
=========================================

Help using this script is available at the command line:

    $ lvbp_run.py --help

Now we demonstrate a simple example for `lvbp_run.py` using the following commands::

    cd example_data
    lvbp_run.py -expt measurements.dat -sigma uncertainties.dat \
    -pred predictions.npz -out test.h5 -num 25000 -reg 5.0 -thin 25 \
    -pop pops.dat

This command loads the experimental data saved in ``measurements.dat``, the 
uncertainties stored in ``uncertainties.dat``, and the predicted experimental
observables in ``predictions.npz``.  In FitEnsemble, we assume that ``.dat`` 
files are arrays readable by ``numpy.loadtxt()``, while ``.npz`` files
are arrays readable by ``numpy.load()``.

The resulting Markov chain Monte Carlo (MCMC) trace will be stored in ``test.h5``.  
The LVBP model will have a regularization
strength of 5.0.  PyMC will perform 25,000 steps of MCMC and subsample the 
resulting trace by 25X.  Finally, the most probably (maximum a posteriori) 
conformational populations will be saved in ``pops.dat``.  
