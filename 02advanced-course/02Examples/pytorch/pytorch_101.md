# Introduction to PyTorch

Based on Bourkes's https://www.learnpytorch.io/

## 1. Tensors are everywhere

Tensors are the fundamental building blocks of machine learning.


```python
import torch
torch.__version__
```




    '1.13.1'



Start by creating basic tensors

- scalar
- vector
- matrix
- tensor

Please see official doc at https://pytorch.org/docs/stable/tensors.html


```python
# scalar
scalar = torch.tensor(7)
scalar
```




    tensor(7)




```python
scalar.ndim
```




    0



To retrieve the value, use the `item` method


```python
scalar.item()
```




    7




```python
# vector
vector = torch.tensor([7, 7])
vector
```




    tensor([7, 7])




```python
vector.ndim
```




    1




```python
vector.shape
```




    torch.Size([2])



- `ndim` gives the number of external square brackets
- `shape` gives the actual dimension = length


```python
# matrix
MAT = torch.tensor([[7, 8], [9, 10]])
MAT
```




    tensor([[ 7,  8],
            [ 9, 10]])




```python
MAT.ndim
```




    2




```python
MAT.shape
```




    torch.Size([2, 2])




```python
# tensor
TEN = torch.tensor([[[1, 2, 3],
         [3, 6, 9],
         [2, 4, 5]]])
TEN
```




    tensor([[[1, 2, 3],
             [3, 6, 9],
             [2, 4, 5]]])




```python
TEN.shape
```




    torch.Size([1, 3, 3])




```python
TEN.ndim
```




    3



The dimensions of a tensor go from outer to inner. That means there's 1 dimension of 3 by 3.

![](./00-pytorch-different-tensor-dimensions.png)


## Random-valued  Tensors

Tensors of *random numbers* are very common in ML. They are used evrywhere.


```python
rand_tensor = torch.rand(size=(3, 4))
rand_tensor, rand_tensor.dtype
```




    (tensor([[0.2350, 0.7880, 0.7052, 0.5025],
             [0.5523, 0.6008, 0.9949, 0.4443],
             [0.5769, 0.7505, 0.1360, 0.8363]]),
     torch.float32)



## Other special tensors


```python
zeros = torch.zeros(size=(3, 4))
zeros, zeros.dtype
```




    (tensor([[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]),
     torch.float32)




```python
ones = torch.ones(size=(3, 4))
ones, ones.dtype
```




    (tensor([[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]),
     torch.float32)



Create a range of numbers in a tensor.

`torch.arange(start, end, step)`


```python
zero_to_ten = torch.arange(0,10)
zero_to_ten
```




    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



To get a tensor of the same shape as another.

`torch.zeros_like(input)`


```python
ten_zeros = torch.zeros_like(zero_to_ten)
ten_zeros
```




    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



### Tensor datatypes

Many datatypes are available, some specific for CPUs, others better for GPUs.

Default datatype is a `float32`, defined by `torch.float32()` or just `torch.float()`.

Lower precision values are faster to compute with, but less acccurate...


```python
# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device
```




    (torch.Size([3]), torch.float32, device(type='cpu'))



Let us place a tensor on the GPU (usually "cuda", bit here it is "mps" for a Mac)


```python
# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device="mps", # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device
```




    (torch.Size([3]), torch.float32, device(type='mps', index=0))



Most common issues for mismatch are:

- shape differences
- datatype
- device issues


```python
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

float_16_tensor.dtype
```




    torch.float16



### Get info from tensors

This is often necessary to ensure compatibility and to avoid pesky bugs.


```python
# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU
```

    tensor([[0.7783, 0.1803, 0.1316, 0.2174],
            [0.5707, 0.7213, 0.5195, 0.5730],
            [0.6286, 0.9001, 0.8025, 0.5707]])
    Shape of tensor: torch.Size([3, 4])
    Datatype of tensor: torch.float32
    Device tensor is stored on: cpu


### Tensor operations

- Basic operations
- Matrix multiplication
- Aggregation (min, max, mean, etc.)
- Reshaping, squeezing


```python
# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
tensor + 10
```




    tensor([11, 12, 13])




```python
# Multiply it by 10
tensor * 10
```




    tensor([10, 20, 30])




```python
# Can also use torch functions
tm = torch.mul(tensor, 10)
ta = torch.add(tensor, 10)

print("tm = ", tm)
print("ta = ", ta)
```

    tm =  tensor([10, 20, 30])
    ta =  tensor([11, 12, 13])



```python
# Element-wise multiplication 
# (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)
```

    tensor([1, 2, 3]) * tensor([1, 2, 3])
    Equals: tensor([1, 4, 9])



```python
tensor = torch.tensor([1, 2, 3])
# Element-wise matrix multiplication
tensor * tensor
```




    tensor([1, 4, 9])




```python
# Matrix multiplication
torch.matmul(tensor, tensor)
```




    tensor(14)



Built-in `torch.matmul()` is much faster and should always be used.


```python
%%time
# Matrix multiplication by hand 
# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
value
```

    CPU times: user 1.16 ms, sys: 738 µs, total: 1.89 ms
    Wall time: 1.31 ms





    tensor(14)




```python
%%time
torch.matmul(tensor, tensor)
```

    CPU times: user 310 µs, sys: 85 µs, total: 395 µs
    Wall time: 368 µs





    tensor(14)



Of course, shapes must be compatible for matrix multiplication...


```python
# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

torch.matmul(tensor_A, tensor_B) # (this will error)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Input In [7], in <cell line: 10>()
          2 tensor_A = torch.tensor([[1, 2],
          3                          [3, 4],
          4                          [5, 6]], dtype=torch.float32)
          6 tensor_B = torch.tensor([[7, 10],
          7                          [8, 11], 
          8                          [9, 12]], dtype=torch.float32)
    ---> 10 torch.matmul(tensor_A, tensor_B)


    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)



```python
# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T)
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.]])
    tensor([[ 7.,  8.,  9.],
            [10., 11., 12.]])



```python
# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")
```

    Original shapes: tensor_A = torch.Size([3, 2]), tensor_B = torch.Size([3, 2])
    
    New shapes: tensor_A = torch.Size([3, 2]) (same as above), tensor_B.T = torch.Size([2, 3])
    
    Multiplying: torch.Size([3, 2]) * torch.Size([2, 3]) <- inner dimensions match
    
    Output:
    
    tensor([[ 27.,  30.,  33.],
            [ 61.,  68.,  75.],
            [ 95., 106., 117.]])
    
    Output shape: torch.Size([3, 3])



```python
# torch.mm is a shortcut for matmul
torch.mm(tensor_A, tensor_B.T)
```




    tensor([[ 27.,  30.,  33.],
            [ 61.,  68.,  75.],
            [ 95., 106., 117.]])



## Tensor multiplication: example of linear regression

Neural networks are full of matrix multiplications and dot products.

The `torch.nn.Linear()` module (that we will see in the next pytorch tutorial), also known as a feed-forward layer or fully connected layer, implements a matrix multiplication between an input $x$ and a weights matrix $A.$

$$ y = x A^T + b, $$

where

- $x$ is the input to the layer (deep learning is a stack of layers like torch.nn.Linear() and others on top of each other).
- $A$ is the weights matrix created by the layer, this starts out as random numbers that get adjusted as a neural network learns to better represent patterns in the data (notice the "T", that's because the weights matrix gets transposed). Note: You might also often see $W$ or another letter like $X$ used to denote the weights matrix.
- $b$ is the bias term used to slightly offset the weights and inputs.
- $y$  is the output (a manipulation of the input in the hope to discover patterns in it).

This is just a linear function, of type $y = ax+b,$ that can be used to draw a straight line.




```python
# Since the linear layer starts with a random weights matrix, we make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2,  # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
```

    Input shape: torch.Size([3, 2])
    
    Output:
    tensor([[2.2368, 1.2292, 0.4714, 0.3864, 0.1309, 0.9838],
            [4.4919, 2.1970, 0.4469, 0.5285, 0.3401, 2.4777],
            [6.7469, 3.1648, 0.4224, 0.6705, 0.5493, 3.9716]],
           grad_fn=<AddmmBackward0>)
    
    Output shape: torch.Size([3, 6])


## Aggregation Functions

Now for some aggregation functions.


```python
# Create a tensor
x = torch.arange(0, 100, 10)
x
```




    tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])




```python
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
```

    Minimum: 0
    Maximum: 90
    Mean: 45.0
    Sum: 450



```python
# alternative: use torch methods
torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)
```




    (tensor(90), tensor(0), tensor(45.), tensor(450))



Positional min/max functions are 

- `torch.argmin()`
- `torch.argmax()` 


```python
# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
```

    Tensor: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
    Index where max value occurs: 8
    Index where min value occurs: 0


Changing datatype is possible (recasting)


```python
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
tensor.dtype
```




    torch.float32




```python
# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
tensor_float16
```




    tensor([10., 20., 30., 40., 50., 60., 70., 80., 90.], dtype=torch.float16)




```python
# Create a int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8
```




    tensor([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=torch.int8)



### Reshaping, stacking, squeezing of tensors

- `torch.reshape(input, shape)`
- `torch.Tensor.view(shape)` to obtain a view
- `torch.stack(tensors, dim=0)` to concatenate along a given direction
- `torch.squeeze(input)` to remove all dimensions of value `1`
- `torch.unsqueeze(input, dim)` to add a dimension of value `1` at `dim`
- `torch.permute(input, dims)` to permute to `dims`


```python
# Create a tensor
import torch
x = torch.arange(1., 8.)
x, x.shape
```




    (tensor([1., 2., 3., 4., 5., 6., 7.]), torch.Size([7]))




```python
# Add an extra dimension
x_reshaped = x.reshape(1, 7)
x_reshaped, x_reshaped.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))




```python
# Change view (keeps same data as original but changes view)
z = x.view(1, 7)
z, z.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))




```python
# Changing z changes x
z[:, 0] = 5
z, x
```




    (tensor([[5., 2., 3., 4., 5., 6., 7.]]), tensor([5., 2., 3., 4., 5., 6., 7.]))




```python
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
x_stacked
```




    tensor([[5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.]])




```python
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=1) # try changing dim to dim=1 and see what happens
x_stacked
```




    tensor([[5., 5., 5., 5.],
            [2., 2., 2., 2.],
            [3., 3., 3., 3.],
            [4., 4., 4., 4.],
            [5., 5., 5., 5.],
            [6., 6., 6., 6.],
            [7., 7., 7., 7.]])




```python
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
```

    Previous tensor: tensor([[5., 2., 3., 4., 5., 6., 7.]])
    Previous shape: torch.Size([1, 7])
    
    New tensor: tensor([5., 2., 3., 4., 5., 6., 7.])
    New shape: torch.Size([7])



```python
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
```

    Previous shape: torch.Size([224, 224, 3])
    New shape: torch.Size([3, 224, 224])


## Slicing

Often we need to extract subsets of data from tensors, usually, some rows or columns.

Let's look at indexing.


```python
# Create a tensor 
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
```




    (tensor([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]]),
     torch.Size([1, 3, 3]))




```python
# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}")
```

    First square bracket:
    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    Second square bracket: tensor([1, 2, 3])
    Third square bracket: 1



```python
# Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension
x[:, :, 1]
```




    tensor([[2, 5, 8]])



### PyTorch tensors and NumPy

We often need to interact with `numpy`, especially for numerical computations. 

The two main methods are:

- `torch.from_numpy(ndarray)`
- `torch.Tensor.numpy()` 


```python
# NumPy array to tensor
import torch
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
array, tensor
```




    (array([1., 2., 3., 4., 5., 6., 7.]),
     tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))




```python
# many pytorch calculations require 'float32'
tensor32 = torch.from_numpy(array).type(torch.float32)
tensor32, tensor32.dtype
```




    (tensor([1., 2., 3., 4., 5., 6., 7.]), torch.float32)




```python
# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
tensor, numpy_tensor
```




    (tensor([1., 1., 1., 1., 1., 1., 1.]),
     array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))



## 2. Reproducibility

To ensure reproducibility of computations, especially ML training,  we need to set the random seed.


```python
import torch
#import random

# # Set the random seed
RANDOM_SEED=42 # try changing this to different values and see what happens to the numbers below
torch.manual_seed(seed=RANDOM_SEED) 
random_tensor_C = torch.rand(3, 4)

# Have to reset the seed every time a new rand() is called 
# Without this, tensor_D would be different to tensor_C 
torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
random_tensor_C == random_tensor_D
```

    Tensor C:
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    
    Tensor D:
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    
    Does Tensor C equal Tensor D? (anywhere)





    tensor([[True, True, True, True],
            [True, True, True, True],
            [True, True, True, True]])



## 3. Running computations on the GPU

See also the code in `pytorch_M2.ipynb`.

Usually the command sequence is:

```python
# Check for GPU
import torch
torch.cuda.is_available()
# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```


```python
# check for gpu on mac
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device
```




    'mps'



To put tensors on the GPU, just use the method `tensor.to(device)`, or put the device option directly into the tensor initialization as seen above:

```python
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  
                               device="mps",  
                               requires_grad=False)  
```


```python
# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu
```

    tensor([1, 2, 3]) cpu





    tensor([1, 2, 3], device='mps:0')



If you need to interact with your tensors (numpy, matplotlib, etc.), then need to get them back to the CPU. Here we use the method `Tensor.cpu()`


```python
# If tensor is on GPU, can't transform it to NumPy (this will error)
tensor_on_gpu.numpy()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Input In [72], in <cell line: 2>()
          1 # If tensor is on GPU, can't transform it to NumPy (this will error)
    ----> 2 tensor_on_gpu.numpy()


    TypeError: can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.



```python
# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu
```




    array([1, 2, 3])




```python
# original is still on the GPU
tensor_on_gpu
```




    tensor([1, 2, 3], device='mps:0')




```python

```
