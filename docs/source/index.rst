.. FitEnsemble documentation master file, created by
   sphinx-quickstart on Sat Mar 30 20:35:16 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FitEnsemble
============

:Release: |version|
:Date: |today|



`FitEnsemble` is a software package for modeling conformation ensembles of 
macromolecules.  Using the linear virtual biasing potential (LVBP) formalism [R1]_, FitEnsemble infers 
conformational ensembles that best capture a set of experimental
measurements.  

To construct a conformational ensemble, you will need a "prior" ensemble.
This guess should ideally be sampled from the approximate equilibrium
of your system.  To perform an LVBP calculation, you will need three inputs:

 * A set of conformations (from MD simulations, for example)

 * A set of experimental measurements and their uncertainties 

 * Predicted experimental observables for each conformation 

Table of Contents:
==================

.. toctree::
   :maxdepth: 1

   installation
   tutorial
   command_line


.. [R1] Beauchamp, K. A., Das, R. , Pande, V. S.  Inferring Conformational 
	Ensembles from Noisy Experiments.  In preparation.  
