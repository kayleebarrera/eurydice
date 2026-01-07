eurydice
========

.. raw:: html

   <style>
     h1:first-of-type {
       display: none;
     }
   </style>

.. image:: eurydice_logo_text.png
   :width: 300px
   :height: 100px
   :align: center

.. raw:: html

   <div style="margin-top: 20px;"></div>

.. admonition:: AAS 247 Note

   (👋 If you're here from my iPoster at AAS, thanks for stopping by!)

   My paper on ``eurydice`` is currently in review for publication in The Astronomical Journal 🎉 - I'm currently working to refurbish some code, add documenation where needed, and set up some tutorials in the meantime. Stay tuned!

   Any particular suggestions or feedback? Feel free to reach out!

Hello there! Welcome to the documentation for ``eurydice``, a Python package for performing cross-validation on Gaussian Process models of radial velocity data. 

``eurydice`` is actively being developed in a `public repository on GitHub <https://github.com/kayleebarrera/eurydice>`_. If you find a bug, want to request a feature, etc. please
create an `issue on GitHub <https://github.com/kayleebarrera/eurydice/issues>`_.  


User Guide
+++++++++++

.. toctree::
   :maxdepth: 2

   installation 
   tutorials
   api


Changelog
++++++++++

**0.5 (2026-1-06)**
She's back!

- added input module to easily format data for eurydice's CV method
- added QuasiPeriodic and SquaredExponential kernels to kernel module
- adjusted a lot of CrossValiation's methods to make it easier to use
- updated plotting methods for more customization and an easy multipanel plot for CV results

**0.4 (2024-8-16)**

- restructured CrossValidation object to take in pre-defined training and test sets 
- restructered split function and CrossValidation object to take in data as pandas dataframes for compactness
- new Kernel module: created a Kernel abstract base class for users to utilize in defining their kernels and restructured build_covariance_matrix function and defaultKernel 
- adding/modifying unit tests for new restructured code


**0.3 (2024-7-24)**

- officially named eurydice!!
- implementation of first version of docs
- added first set of test functions
