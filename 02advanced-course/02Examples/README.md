# Advanced Course Examples


## PyTorch

There are many tutorials and examples available for learning PyTorch. Here are some basics.

- Basics $\rightarrow$ [notebook](pytorch/pytorch_101.ipynb), [html](pytorch/pytorch_101.html), [markdown](pytorch/pytorch_101.md)
- Neural network basics and ML workflows $\rightarrow$ [notebook](pytorch/pytorch_102.ipynb), [html](pytorch/pytorch_102.html), [markdown](pytorch/pytorch_102/pytorch_102.md)
- Use of GPUs in PyTorch $\rightarrow$ [notebook](pytorch/Torch_test_GPU_CPU.ipynb), [html](pytorch/Torch_test_GPU_CPU.html), [markdown](pytorch/Torch_test_GPU_CPU.md)


In the sections below, the reader will find PyTorch implementations of: 

- Optimization
- Machine Learning
- SciML


## Optimization and AD

- Unconstrained optimization
   - Comparison of gradient descent and stochastic gradient $\rightarrow$ [notebook](opt/GDvsSGD.ipynb), [html](opt/GDvsSGD.html)
   - Comparison of different algorithms on a non-convex, 2D problem $\rightarrow$ [notebook](opt/opt_himmelblau.ipynb), [html](opt/opt_himmelblau.html), [pdf](opt/opt_himmelblau.pdf)
   - Comparison of different initial guesses for a non-convex, 2D problem, with animation $\rightarrow$ [notebook](opt/opt_visu.ipynb), [html](opt/opt_visu.html),  [pdf](opt/opt_visu.pdf)
   - PyTorch for 1D optimization problems $\rightarrow$ [notebook](opt/torch-opt-simplest.ipynb), [html](opt/torch-opt-simplest.html),  [pdf](opt/torch-opt-simplest.pdf)
   - Pytorch detailed comparison and diagnostics of SGD and LBFGS methods $\rightarrow$ [notebook](opt/torch_lbfgs_convergence.ipynb), [html](opt/torch_lbfgs_convergence.html),  [pdf](opt/torch_lbfgs_convergence.pdf)
   - Pytorch linear regression curve fitting by least-squares minimization $\rightarrow$ [notebook](opt/torch_linreg_basic.ipynb), [html](opt/ttorch_linreg_basic.html),  [pdf](opt/torch_linreg_basic.pdf)
   - Pytorch exponential curve fitting by least-squares minimization using Adam $\rightarrow$ [notebook](opt/torch_curve_fitting.ipynb), [html](opt/torch_curve_fitting.html),  [pdf](opt/torch_curve_fitting.pdf)


- Constrained optimization 
   - Quadratic function with equality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](opt/Constrained_opt.ipynb), [markdown](opt/Constrained_opt/Constrained_opt.md), [html](opt/Constrained_opt.html)
   - Quadratic function with inequality constraint using Scipy's `minimize` function $\rightarrow$ [notebook](opt/Constrained_inequality.ipynb), [markdown](opt/Constrained_inequality/Constrained_inequality.md), [html](opt/Constrained_inequality.html)


- Introduction to differentiable programming $\rightarrow$ [notebook]ad/diff_prog.ipynb), [markdown](ad/diff_prog/diff_prog.md), [html](ad/diff_prog.html), [pdf](ad/diff_prog.pdf)
- Simple linear regression with `autograd`  $\rightarrow$ [notebook](ad/autograd_lin_reg.ipynb),  [html](ad/autograd_lin_reg.html), [pdf](ad/autograd_lin_reg.pdf)
- `autograd` tutorial  $\rightarrow$ [notebook](ad/autograd_tut.ipynb), [markdown](ad/autograd_tut/autograd_tut.md), [html](ad/autograd_tut.html)

## Machine Learning for SciML

- Linear regression tutorial with PyTorch, numpy and sklearn comparisons $\rightarrow$ [notebook](linreg/torch_linreg_tutorial.ipynb), [markdown](linreg/torch_linreg_tutorial/torch_linreg_tutorial.md), [html](linreg/torch_linreg_tutorial.html)

- Simple NN classification with Pytorch $\rightarrow$ [notebook](ml/torch_NN_class_simple.ipynb), [markdown](02Examples/ml/torch_NN_class_simple/.md), [html](ml/torch_NN_class_simple.html)


- NN regression on socio-economic housing data $\rightarrow$ [notebook](ml/pytorch_NN_reg.ipynb), [markdown](pytorch_NN_reg/pytorch_NN_reg/.md), [html](ml/pytorch_NN_reg.html)


- NN classification on diabetes clinical data  $\rightarrow$ [notebook](ml/pytorch_NN_classif.ipynb), [markdown](ml/pytorch_NN_classif.md), [html](ml/pytorch_NN_classif.html), [pdf](ml/pytorch_NN_classif.pdf)


- Cross-validation and Tuning
   - Precision-Recall curve for heart disease data  $\rightarrow$ [notebook](ml/ML_prec_recall.ipynb), [markdown](ml/ML_prec_recall/ML_prec_recall/ML_prec_recall.md), [html](ml/ML_prec_recall.html)
   - see also the numerous Basic Course Examples


## Surrogate Modeling (SUMO)

- Linear regression for cyclical features $\rightarrow$ [notebook](SUMO/cyclic_data.ipynb), [markdown](SUMO/cyclic_data/cyclic_data.md)

- Multilinear regression with categorical variables for concrete strength data  $\rightarrow$ [notebook](SUMO/mlreg_concrete.ipynb), [markdown](SUMO/mlreg_concrete/mlreg_concrete.md)

- Nonlinear regression using SVM for LIDAR data [markdown](https://github.com/markasch/CSU-IMU-2023/blob/main/01basic-course/02Examples/svm_reg/svm_reg.md) [Rmd](https://github.com/markasch/CSU-IMU-2023/blob/main/01basic-course/02Examples/svm_reg/svm_reg.Rmd) 

## SciML

- Learn a sine function $\rightarrow$ [notebook](PINN/pytorch_NN_fct_approx.ipynb), [markdown](PINN/pytorch_NN_fct_approx/pytorch_NN_fct_approx.md), [html](PINN/pytorch_NN_fct_approx.html), [pdf](PINN/pytorch_NN_fct_approx.pdf)

- Learn a stepwise function $\rightarrow$ [notebook](PINN/pytorch_NN_single_layer.ipynb), [markdown](PINN/pytorch_NN_single_layer/pytorch_NN_single_layer.md), [html](PINN/pytorch_NN_single_layer.html), [pdf](PINN/pytorch_NN_single_layer.pdf)

- Constant parameter identification inverse problem for a (P)DE  $\rightarrow$ [notebook](PINN/PCL_1D_param_const.ipynb), [markdown](PINN/PCL_1D_param_const/PCL_1D_param_const.md), [html](PINN/PCL_1D_param_const.html), [pdf](PINN/PCL_1D_param_const.pdf)


- Function identification in a (P)DE - coming soon...

- PINN for harmonic oscillator  $\rightarrow$ [notebook](PINN/PINN_harmonic.ipynb), [markdown](PINN/PINN_harmonic/PINN_harmonic.md), [html](PINN/PINN_harmonic.html), [pdf](PINN/PINN_harmonic.pdf)


## DeepXDE examples

- Learn a scalar sine function $\rightarrow$ [notebook](DDE/dde_fct_learning.ipynb), [markdown](DDE/dde_fct_learning/dde_fct_learning.md), [html](DDE/dde_fct_learning.html), [pdf](DDE/dde_fct_learning.pdf)


- Learn the solution to a very simple system of ODEs  $\rightarrow$ [notebook](DDE/dde_simple.ipynb), [markdown](DDE/dde_simple/dde_simple.md), [html](DDE/dde_simple.html), [pdf](DDE/dde_simple.pdf)


- Learn the solution to the Lotka-Volterra system of ODEs  $\rightarrow$ [notebook](DDE/dde_LK.ipynb), [markdown](DDE/dde_LK/dde_LK.md), [html](DDE/dde_LK.html), [pdf](DDE/dde_LK.pdf)

- Learn the solution to the 1D Poisson equation with Dirichlet and Robin BCs  $\rightarrow$ [notebook](DDE/dde_Poisson1D.ipynb), [markdown](DDE/dde_Poisson1D/dde_Poisson1D.md), [html](DDE/dde_Poisson1D.html), [pdf](DDE/dde_Poisson1D.pdf)

- Learn the solution to the 2D Poisson equation on an L-shaped domain  $\rightarrow$ [notebook](DDE/dde_Poisson2D.ipynb), [markdown](DDE/dde_Poisson2D/dde_Poisson2D.md), [html](DDE/dde_Poisson2D.html), [pdf](DDE/dde_Poisson2D.pdf)

## Data Assimilation


### Variational DA

Variational DA, based on adjoint methods, is complicated to code. We recommend to rather use the reliable open-source code repositories that are referenced in the DA [Lecture](https://github.com/markasch/CSU-IMU-2023/blob/main/01basic-course/01Lectures/12_DA_var.pdf). We provide a single example here, 3D Var,  since it is relatively straightforward to code.

- 3D Var for Lorenz63 system $\rightarrow$ [notebook](DA/threeDVar_l63.ipynb), [markdown](DA/threeDVar/threeDVar.md), [html](DA/threeDVar.html)


### Statistical DA

Statistical DA is less complex to code, in particular in the form of simple Kalman Filters. Extensions of the KF are studied in the [Advanced Course](https://sites.google.com/view/csu2023/advanced-course), which should be consulted for more details and examples. 

- Kalman Filter for a scalar, Gaussian random walk  $\rightarrow$ [notebook](DA/kf_gaussRW.ipynb), [markdown](DA/kf_gaussRW/kf_gaussRW.md), [html](DA/kf_gaussRW.html)

- Ensemble Kalman Filter for Lorenz63 system  $\rightarrow$ [notebook](DA/enkf_l63.ipynb), [markdown](DA/enkf_l63/enkf_l63.md), [html](DA/enkf_l63.html)

- Kalman Filter and Extended Kalman Filter for a noisy pendulum with tracking of the position $\rightarrow$ [notebook](DA/kf_ekf_pendulum.ipynb), [markdown](DA/kf_ekf_pendulum/kf_ekf_pendulum.md), [html](DA/kf_ekf_pendulum.html)

- Extended Kalman Filter for a noisy pendulum with tracking of both position and velocity  $\rightarrow$ [notebook](DA/ekf_2D_pendulum.ipynb), [markdown](DA/ekf_2D_pendulum/ekf_2D_pendulum.md), [html](DA/ekf_2D_pendulum.html), [pdf](DA/ekf_2D_pendulum.pdf)





