u"""
setup.py: Install fit_ensemble.
"""
import setuptools

metadata = {
    "name" : "fit_ensemble",
    'version': "1.0.0",
    'author': "Kyle A. Beauchamp",
    'author_email': 'kyleabeauchamp@gmail.com',
    'license': 'GPL v3.0',
    'url': 'https://github.com/kyleabeauchamp/fit_ensemble',
    'download_url': 'https://github.com/kyleabeauchamp/fit_ensemble',
    'install_requires': ['scipy', 'numpy', 'pymc', 'tables'],
    'platforms': ["Linux", "Mac OS X"],
    "packages" : ["fit_ensemble"],
    "package_dir" : {"fit_ensemble" : "src/"},
    #"package_data" : {"src": ["test.txt"]},  # This doesn't work because of a character deletion bug in setuptools.
    "data_files" : [("fit_ensemble/example_data", ["example_data/conf.pdb", "example_data/trajout.xtc", "example_data/rama.npz"])],
    "include_package_data" : True,
    'description': "Python code for inferring conformational ensembles.",
    'long_description': """fit_ensemble  (https://github.com/kyleabeauchamp/fit_ensemble)
    is a library that allows scientists to combine simulation and experimental 
    data to infer the conformational ensemble of a protein."""
}

if __name__ == '__main__':
    setuptools.setup(**metadata)
