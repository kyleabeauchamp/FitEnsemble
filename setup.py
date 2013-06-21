u"""
setup.py: Install FitEnsemble.
"""
import setuptools

metadata = {
    "name" : "FitEnsemble",
    'version': "1.0.0",
    'author': "Kyle A. Beauchamp",
    'author_email': 'kyleabeauchamp@gmail.com',
    'license': 'GPL v3.0',
    'url': 'https://github.com/kyleabeauchamp/FitEnsemble',
    'download_url': 'https://github.com/kyleabeauchamp/FitEnsemble',
    'install_requires': ['scipy', 'numpy', 'pymc', 'tables'],
    'platforms': ["Linux", "Mac OS X"],
    "scripts" : [],
    "packages" : ["fitensemble", "fitensemble.nmr_tools"],
    "package_dir" : {"fitensemble" : "fitensemble/"},
    "data_files" : [("fitensemble/example_data", ["example_data/conf.pdb", "example_data/trajout.xtc", "example_data/rama.npz", "example_data/experiments.tab"])],
    "include_package_data" : True,
    'description': "Python code for inferring conformational ensembles.",
    'long_description': """FitEnsemble  (https://github.com/kyleabeauchamp/FitEnsemble)
    is a library that allows scientists to combine simulation and experimental 
    data to infer the conformational ensemble of a protein."""
}

if __name__ == '__main__':
    setuptools.setup(**metadata)
