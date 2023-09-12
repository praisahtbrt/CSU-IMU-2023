# Advanced Course Examples


## PyTorch

There are many tutorials and examples available for learning PyTorch. 

- Basics $\rightarrow$ [notebook](02Examples/pytorch/pytorch_101.ipynb), [html](02Examples/pytorch/pytorch_101.html), [markdown](02Examples/pytorch/pytorch_101.md)
- Neural network basics and ML workflows $\rightarrow$ [notebook](02Examples/pytorch/pytorch_102.ipynb), [html](02Examples/pytorch/pytorch_102.html), [markdown](02Examples/pytorch/pytorch_102/pytorch_102.md)


In the sections below, the reader will find PyTorch implementations of: 

- optimization
- Machine Learning
- SciML


## Optimization and AD

- Unconstrained optimization
   - Comparison of gradient descent and stochastic gradient $\rightarrow$ [notebook](02Examples/opt/GDvsSGD.ipynb), [html](02Examples/opt/GDvsSGD.html)
   - Comparison of different algorithms on a non-convex, 2D problem $\rightarrow$ [notebook](02Examples/opt/opt_himmelblau.ipynb), [html](02Examples/opt/opt_himmelblau.html), [pdf](02Examples/opt/opt_himmelblau.pdf)
   - Comparison of different initial guesses for a non-convex, 2D problem, with animation $\rightarrow$ [notebook](02Examples/opt/opt_visu.ipynb), [html](02Examples/opt/opt_visu.html),  [pdf](02Examples/opt/opt_visu.pdf)


- Constrained optimization 
   - Quadratic function with equality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](02Examples/opt/Constrained_opt.ipynb), [markdown](02Examples/opt/Constrained_opt/Constrained_opt.md), [html](02Examples/opt/Constrained_opt.html)
   - Quadratic function with inequality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](02Examples/opt/Constrained_inequality.ipynb), [markdown](02Examples/opt/Constrained_inequality/Constrained_inequality.md), [html](02Examples/opt/Constrained_inequality.html)


- Introduction to differentiable programming $\rightarrow$ [notebook](02Examples/ad/diff_prog.ipynb), [markdown](02Examples/ad/diff_prog/diff_prog.md), [html](02Examples/ad/diff_prog.html), [pdf](02Examples/ad/diff_prog.pdf)
- Simple linear regression with `autograd`  $\rightarrow$ [notebook](02Examples/ad/autograd_lin_reg.ipynb),  [html](02Examples/ad/autograd_lin_reg.html), [pdf](02Examples/ad/autograd_lin_reg.pdf)
- `autograd` tutorial  $\rightarrow$ [notebook](02Examples/ad/autograd_tut.ipynb), [markdown](02Examples/ad/autograd_tut/autograd_tut.md), [html](02Examples/ad/autograd_tut.html)

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