# Advanced Course Examples


## PyTorch

There are many tutorials and examples available for learning PyTorch. 

- Basics 
- Neural network basics and ML workflows


In the sections below, the reader will find PyTorch implementations of: 

- optimization
- Machine Learning
- SciML


## Optimization and AD

- Unconstrained optimization
   - Comparison of gradient descent and stochastic gradient $\rightarrow$ [notebook](opt/GDvsSGD.ipynb), [html](opt/GDvsSGD.html)
   - Comparison of different algorithms on a non-convex, 2D problem $\rightarrow$ [notebook](opt/opt_himmelblau.ipynb), [html](opt/opt_himmelblau.html), [pdf](opt/opt_himmelblau.pdf)
   - Comparison of different initial guesses for a non-convex, 2D problem, with animation $\rightarrow$ [notebook](opt/opt_visu.ipynb), [html](opt/opt_visu.html),  [pdf](opt/opt_visu.pdf)


- Constrained optimization 
   - Quadratic function with equality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](.ipynb), [markdown](.md), [html](.html)
   - Quadratic function with inequality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](.ipynb), [markdown](.md), [html](.html)


- Introduction to differentiable programming $\rightarrow$ [notebook](ad/diff_prog.ipynb), [markdown](ad/diff_prog/diff_prog.md), [html](ad/diff_prog.html), [pdf](ad/diff_prog.pdf)
- Simple linear regression with `autograd`  $\rightarrow$ [notebook](autograd_lin_reg.ipynb),  [html](autograd_lin_reg.html), [pdf](ad/autograd_linreg.pdf)
- `autograd` tutorial  $\rightarrow$ [notebook](ad/autograd_tut.ipynb), [markdown](ad/autograd_tut/autograd_tut.md), [html](ad/autograd_tut.html), [pdf](ad/autograd_tut.pdf)

## Machine Learning for SciML

- Linear regression (PyTorch)

- Cross-validation and Tuning
   - Precision-Recall curve for heart disease data  $\rightarrow$ [notebook](.ipynb), [markdown](.md), [html](.html)
   - see also the Basic Course Examples


## SciML

- PINN for harmonic oscillator


## Data Assimilation


### Variational DA

Variational DA, based on adjoint methods, is complicated to code. We recommend to rather use the reliable open-source code repositories that are referenced in the DA [Lecture](https://github.com/markasch/CSU-IMU-2023/blob/main/01basic-course/01Lectures/12_DA_var.pdf).


### Statistical DA

Statistical DA is less complex to code, in particular in the form of simple Kalman Filters. Extensions of the KF will be studied in the [Advanced Course](https://sites.google.com/view/csu2023/advanced-course), which should be consulted for more details and examples.