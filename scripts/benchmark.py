import numpy as np
import matplotlib.pyplot as plt
from fitensemble import lvbp, example_loader

lvbp.ne.set_num_threads(5)

predictions, measurements, uncertainties = example_loader.load_alanine_numpy()
num_samples = 20000  # Generate 20,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

regularization_strength = 3.0  

lvbp_model = lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength)

%prun lvbp_model.sample(num_samples, thin=thin, burn=burn)

"""
OMP 1, NT 1
         2444779 function calls (2161779 primitive calls) in 92.518 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20375   22.425    0.001   48.867    0.002 lvbp.py:47(get_populations_from_q)
    80750   21.023    0.000   21.023    0.000 {method 'dot' of 'numpy.ndarray' objects}
    20375   20.237    0.001   20.786    0.001 necompiler.py:667(evaluate)
    80752   14.925    0.000   14.925    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20375    8.374    0.000   23.153    0.001 lvbp.py:23(get_q)
241351/120601    0.447    0.000   90.824    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
121351/80601    0.438    0.000   77.224    0.001 PyMCObjects.py:434(get_value)
    20000    0.358    0.000    0.895    0.000 ensemble_fitter.py:36(get_chi2)
    20000    0.300    0.000   90.868    0.005 StepMethods.py:434(step)
    20000    0.287    0.000   12.117    0.001 lvbp.py:211(logp_prior)
    20375    0.276    0.000    0.309    0.000 necompiler.py:462(getContext)
121126/60376    0.233    0.000   77.306    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20375    0.231    0.000    5.635    0.000 _methods.py:42(_mean)

OMP 2, NT 2

         2444896 function calls (2161888 primitive calls) in 87.676 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20376   23.106    0.001   43.871    0.002 lvbp.py:47(get_populations_from_q)
    80752   20.112    0.000   20.112    0.000 {method 'dot' of 'numpy.ndarray' objects}
    80754   15.413    0.000   15.413    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20376   14.276    0.001   14.881    0.001 necompiler.py:667(evaluate)
    20376    8.695    0.000   24.772    0.001 lvbp.py:23(get_q)
241353/120601    0.486    0.000   85.689    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
121353/80601    0.455    0.000   73.945    0.001 PyMCObjects.py:434(get_value)
    20000    0.403    0.000   85.996    0.004 StepMethods.py:434(step)
    20000    0.370    0.000    0.847    0.000 ensemble_fitter.py:36(get_chi2)
    20376    0.301    0.000    0.337    0.000 necompiler.py:462(getContext)
    20376    0.273    0.000    5.986    0.000 _methods.py:42(_mean)
121129/60377    0.250    0.000   74.001    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
        1    0.229    0.229   87.674   87.674 MCMC.py:252(_loop)

OMP 3, NT 3

         2444557 function calls (2161581 primitive calls) in 85.445 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20372   23.314    0.001   43.159    0.002 lvbp.py:47(get_populations_from_q)
    80744   17.914    0.000   17.914    0.000 {method 'dot' of 'numpy.ndarray' objects}
    80746   15.846    0.000   15.846    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20372   13.230    0.001   13.829    0.001 necompiler.py:667(evaluate)
    20372    9.040    0.000   23.575    0.001 lvbp.py:23(get_q)
241345/120601    0.507    0.000   83.571    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
121345/80601    0.481    0.000   71.410    0.001 PyMCObjects.py:434(get_value)
    20000    0.384    0.000    0.965    0.000 ensemble_fitter.py:36(get_chi2)
    20000    0.331    0.000   83.836    0.004 StepMethods.py:434(step)
    20000    0.325    0.000   10.547    0.001 lvbp.py:211(logp_prior)
    20372    0.291    0.000    0.328    0.000 necompiler.py:462(getContext)
121117/60373    0.254    0.000   71.509    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20372    0.247    0.000    5.872    0.000 _methods.py:42(_mean)
        1    0.223    0.223   85.444   85.444 MCMC.py:252(_loop)
    20000    0.218    0.000    0.581    0.000 linalg.py:1868(norm)
    80000    0.215    0.000   82.323    0.001 PyMCObjects.py:293(get_logp)
    20000    0.199    0.000    0.602    0.000 StepMethods.py:516(propose)
    40000    0.189    0.000   82.729    0.002 Node.py:23(logp_of_set)
    20372    0.185    0.000    0.192    0.000 _methods.py:32(_count_reduce_items)
    20000    0.167    0.000    0.240    0.000 PyMCObjects.py:768(set_value)
    20000    0.148    0.000    0.148    0.000 {method 'normal' of 'mtrand.RandomState' objects}
   101125    0.146    0.000    0.146    0.000 {numpy.core.multiarray.array}
121117/60373    0.145    0.000   71.596    0.001 Container.py:493(get_value)


OMP 4, NT 4

         2445370 function calls (2162274 primitive calls) in 83.638 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20387   23.377    0.001   41.636    0.002 lvbp.py:47(get_populations_from_q)
    80774   17.958    0.000   17.958    0.000 {method 'dot' of 'numpy.ndarray' objects}
    80776   15.437    0.000   15.437    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20387   11.923    0.001   12.518    0.001 necompiler.py:667(evaluate)
    20387    9.190    0.000   24.699    0.001 lvbp.py:23(get_q)
241375/120601    0.468    0.000   81.701    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
121375/80601    0.389    0.000   70.898    0.001 PyMCObjects.py:434(get_value)
    20000    0.385    0.000   81.987    0.004 StepMethods.py:434(step)
    20000    0.337    0.000    0.786    0.000 ensemble_fitter.py:36(get_chi2)
    20387    0.294    0.000    0.329    0.000 necompiler.py:462(getContext)
    20387    0.243    0.000    5.927    0.000 _methods.py:42(_mean)
121162/60388    0.237    0.000   70.962    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
        1    0.225    0.225   83.637   83.637 MCMC.py:252(_loop)
    20000    0.219    0.000    0.640    0.000 StepMethods.py:516(propose)
    20000    0.208    0.000    9.447    0.000 lvbp.py:211(logp_prior)
    80000    0.206    0.000   80.392    0.001 PyMCObjects.py:293(get_logp)
    20000    0.197    0.000    0.449    0.000 linalg.py:1868(norm)
    20000    0.178    0.000    0.259    0.000 PyMCObjects.py:768(set_value)
    20387    0.175    0.000    0.186    0.000 _methods.py:32(_count_reduce_items)
    40000    0.165    0.000   80.785    0.002 Node.py:23(logp_of_set)
    20000    0.147    0.000    0.147    0.000 {method 'normal' of 'mtrand.RandomState' objects}
   101170    0.145    0.000    0.145    0.000 {numpy.core.multiarray.array}
121162/60388    0.143    0.000   71.037    0.001 Container.py:493(get_value)
   182901    0.110    0.000    0.110    0.000 {isinstance}
    20000    0.106    0.000    0.892    0.000 lvbp.py:141(logp)


OMP 5, NT 5

         2449645 function calls (2166285 primitive calls) in 116.273 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20420   35.400    0.002   59.536    0.003 lvbp.py:47(get_populations_from_q)
    80842   23.003    0.000   23.003    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    80840   22.523    0.000   22.523    0.000 {method 'dot' of 'numpy.ndarray' objects}
    20420   14.649    0.001   15.391    0.001 necompiler.py:667(evaluate)
    20420   12.786    0.001   31.881    0.002 lvbp.py:23(get_q)
121441/80601    0.690    0.000   97.244    0.001 PyMCObjects.py:434(get_value)
241441/120601    0.654    0.000  113.922    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    20000    0.517    0.000    1.268    0.000 ensemble_fitter.py:36(get_chi2)
    20000    0.460    0.000   14.593    0.001 lvbp.py:211(logp_prior)
    20000    0.402    0.000  113.894    0.006 StepMethods.py:434(step)
    20420    0.361    0.000    0.407    0.000 necompiler.py:462(getContext)
121261/60421    0.326    0.000   97.373    0.002 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20420    0.323    0.000    8.665    0.000 _methods.py:42(_mean)
    20000    0.284    0.000    0.750    0.000 linalg.py:1868(norm)
    80000    0.281    0.000  111.997    0.001 PyMCObjects.py:293(get_logp)
        1    0.272    0.272  116.271  116.271 MCMC.py:252(_loop)
    20000    0.260    0.000    0.752    0.000 StepMethods.py:516(propose)
    40000    0.245    0.000  112.525    0.003 Node.py:23(logp_of_set)
    20420    0.223    0.000    0.234    0.000 _methods.py:32(_count_reduce_items)
    20000    0.206    0.000    0.301    0.000 PyMCObjects.py:768(set_value)
   101269    0.205    0.000    0.205    0.000 {numpy.core.multiarray.array}
    20000    0.173    0.000    0.173    0.000 {method 'normal' of 'mtrand.RandomState' objects}
121261/60421    0.172    0.000   97.475    0.002 Container.py:493(get_value)
   183066    0.153    0.000    0.153    0.000 {isinstance}



OMP 8, NT 8
         2445310 function calls (2162246 primitive calls) in 111.929 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20383   33.109    0.002   55.556    0.003 lvbp.py:47(get_populations_from_q)
    80766   24.091    0.000   24.091    0.000 {method 'dot' of 'numpy.ndarray' objects}
    80768   21.518    0.000   21.518    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20383   13.454    0.001   14.181    0.001 necompiler.py:667(evaluate)
    20383   11.909    0.001   31.820    0.002 lvbp.py:23(get_q)
241367/120601    0.661    0.000  109.385    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
121367/80601    0.624    0.000   93.591    0.001 PyMCObjects.py:434(get_value)

"""
