# Advanced Course Examples


## PyTorch

There are many tutorials and examples available for learning PyTorch. Here are some basics.

- Basics $\rightarrow$ [notebook](02Examples/pytorch/pytorch_101.ipynb), [html](02Examples/pytorch/pytorch_101.html), [markdown](02Examples/pytorch/pytorch_101.md)
- Neural network basics and ML workflows $\rightarrow$ [notebook](02Examples/pytorch/pytorch_102.ipynb), [html](02Examples/pytorch/pytorch_102.html), [markdown](02Examples/pytorch/pytorch_102/pytorch_102.md)
- Use of GPUs in PyTorch \rightarrow$ [notebook](02Examples/pytorch/Torch_test_GPU_CPU.ipynb), [html](02Examples/pytorch/Torch_test_GPU_CPU.html), [markdown](02Examples/pytorch/Torch_test_GPU_CPU.md)


In the sections below, the reader will find PyTorch implementations of: 

- Optimization
- Machine Learning
- SciML


## Optimization and AD

- Unconstrained optimization
   - Comparison of gradient descent and stochastic gradient $\rightarrow$ [notebook](02Examples/opt/GDvsSGD.ipynb), [html](02Examples/opt/GDvsSGD.html)
   - Comparison of different algorithms on a non-convex, 2D problem $\rightarrow$ [notebook](02Examples/opt/opt_himmelblau.ipynb), [html](02Examples/opt/opt_himmelblau.html), [pdf](02Examples/opt/opt_himmelblau.pdf)
   - Comparison of different initial guesses for a non-convex, 2D problem, with animation $\rightarrow$ [notebook](02Examples/opt/opt_visu.ipynb), [html](02Examples/opt/opt_visu.html),  [pdf](02Examples/opt/opt_visu.pdf)
   - PyTorch for 1D optimization problems $\rightarrow$ [notebook](02Examples/opt/torch-opt-simplest.ipynb), [html](02Examples/opt/torch-opt-simplest.html),  [pdf](02Examples/opt/torch-opt-simplest.pdf)
   - Pytorch detailed comparison and diagnostics of SGD and LBFGS methods $\rightarrow$ [notebook](02Examples/opt/torch_lbfgs_convergence.ipynb), [html](02Examples/opt/torch_lbfgs_convergence.html),  [pdf](02Examples/opt/torch_lbfgs_convergence.pdf)
   - Pytorch linear regression curve fitting by least-squares minimization $\rightarrow$ [notebook](02Examples/opt/torch_linreg_basic.ipynb), [html](02Examples/opt/ttorch_linreg_basic.html),  [pdf](02Examples/opt/torch_linreg_basic.pdf)
   - Pytorch exponential curve fitting by least-squares minimization using Adam $\rightarrow$ [notebook](02Examples/opt/torch_curve_fitting.ipynb), [html](02Examples/opt/torch_curve_fitting.html),  [pdf](02Examples/opt/torch_curve_fitting.pdf)


- Constrained optimization 
   - Quadratic function with equality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](02Examples/opt/Constrained_opt.ipynb), [markdown](02Examples/opt/Constrained_opt/Constrained_opt.md), [html](02Examples/opt/Constrained_opt.html)
   - Quadratic function with inequality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](02Examples/opt/Constrained_inequality.ipynb), [markdown](02Examples/opt/Constrained_inequality/Constrained_inequality.md), [html](02Examples/opt/Constrained_inequality.html)


- Introduction to differentiable programming $\rightarrow$ [notebook](02Examples/ad/diff_prog.ipynb), [markdown](02Examples/ad/diff_prog/diff_prog.md), [html](02Examples/ad/diff_prog.html), [pdf](02Examples/ad/diff_prog.pdf)
- Simple linear regression with `autograd`  $\rightarrow$ [notebook](02Examples/ad/autograd_lin_reg.ipynb),  [html](02Examples/ad/autograd_lin_reg.html), [pdf](02Examples/ad/autograd_lin_reg.pdf)
- `autograd` tutorial  $\rightarrow$ [notebook](02Examples/ad/autograd_tut.ipynb), [markdown](02Examples/ad/autograd_tut/autograd_tut.md), [html](02Examples/ad/autograd_tut.html)

## Machine Learning for SciML

- Linear regression tutorial with PyTorch, numpy and sklearn comparisons $\rightarrow$ [notebook](02Examples/linreg/torch_linreg_tutorial.ipynb), [markdown](02Examples/linreg/torch_linreg_tutorial/torch_linreg_tutorial.md), [html](02Examples/linreg/torch_linreg_tutorial.html)

- Simple NN classification with Pytorch $\rightarrow$ [notebook](02Examples/ml/torch_NN_class_simple.ipynb), [markdown](02Examples/ml/torch_NN_class_simple/.md), [html](02Examples/ml/torch_NN_class_simple.html)


- NN regression on socio-economic housing data $\rightarrow$ [notebook](02Examples/ml/pytorch_NN_reg.ipynb), [markdown](02Examples/ml/pytorch_NN_reg/pytorch_NN_reg/.md), [html](02Examples/ml/pytorch_NN_reg.html)


- NN classification on diabetes data


- Cross-validation and Tuning
   - Precision-Recall curve for heart disease data  $\rightarrow$ [notebook](02Examples/ml/ML_prec_recall.ipynb), [markdown](02Examples/ml/ML_prec_recall/ML_prec_recall/ML_prec_recall.md), [html](02Examples/ml/ML_prec_recall.html)
   - see also the numerous Basic Course Examples


## SciML

- Learn a stepwise function 
- Learn a sine function
- Constant parameter identification in a (P)DE
- Function identification in a (P)DE
- PINN for harmonic oscillator


## Data Assimilation


### Variational DA

Variational DA, based on adjoint methods, is complicated to code. We recommend to rather use the reliable open-source code repositories that are referenced in the DA [Lecture](https://github.com/markasch/CSU-IMU-2023/blob/main/01basic-course/01Lectures/12_DA_var.pdf).


### Statistical DA

Statistical DA is less complex to code, in particular in the form of simple Kalman Filters. Extensions of the KF will be studied in the [Advanced Course](https://sites.google.com/view/csu2023/advanced-course), which should be consulted for more details and examples.