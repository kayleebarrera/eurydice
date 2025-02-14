EURYDICE: A WIP to create a package that will perform cross-validations for radial velocity signal models with Gaussian processes.

For more detailed documentation, start [here](https://eurydice.readthedocs.io/en/latest/)

[![PyPI version](https://badge.fury.io/py/eurydice.svg)](https://badge.fury.io/py/eurydice)
[![Documentation Status](https://readthedocs.org/projects/eurydice/badge/?version=latest)](https://eurydice.readthedocs.io/en/latest/?badge=latest)
![Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Last Updated: August 16 2024

Current Version: 0.4

To-Do:
* adding compatibility with pre-existing GP packages (george, tinygp, celerite): the gist is probably to let code read if the kernel function its passed is a certain object (e.g a george kernel object) and let the code run the GPR and CV using that package's methods
* writing a tutorial on how to use code with juypter notebook
* combing through existing code to improve lacking documentation and find more appropriate places to include error messages
* writing more tests: how to write tests for GP_predict, run_CV and plotting functions? 
* add std dev to histogram legend
