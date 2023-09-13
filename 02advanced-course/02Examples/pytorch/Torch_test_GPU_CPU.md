```python
# check availability of GPU
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

    tensor([1.], device='mps:0')



```python
# toy example on GPU
import timeit
import torch
import random

x = torch.ones(5000, device="mps")
timeit.timeit(lambda: x * random.randint(0,100), number=100000)
#Out[17]: 4.568202124999971

# toy example cpu
#x = torch.ones(5000, device="cpu")
# timeit.timeit(lambda: x * random.randint(0,100), number=100000)
#Out[18]: 0.30446054200001527
```




    2.0122362919998977




```python
# toy example on cpu
x = torch.ones(5000, device="cpu")
timeit.timeit(lambda: x * random.randint(0,100), number=100000)
```




    0.24692429099991386



The CPU is approximately 10 times faster than the GPU...

Here is a slightly more complex examples, with a matrix-vector tensor multiplication.


```python
a_cpu = torch.rand(250, device='cpu')
b_cpu = torch.rand((250, 250), device='cpu')
a_mps = torch.rand(250, device='mps')
b_mps = torch.rand((250, 250), device='mps')

print('cpu', timeit.timeit(lambda: a_cpu @ b_cpu, number=100_000))
print('mps', timeit.timeit(lambda: a_mps @ b_mps, number=100_000))
```

    cpu 0.8405147910000323
    mps 2.3573820419999265


Now, we drastically increase the problem size, using the tensor dimension.


```python
x = torch.ones(50000000, device="mps")
timeit.timeit(lambda: x * random.randint(0,100), number=1)
```




    0.00048149999997804116




```python
x = torch.ones(50000000, device="cpu")
timeit.timeit(lambda: x * random.randint(0,100), number=1)
```




    0.03234533299996656




```python
.0323/.00048
```




    67.29166666666667



## Conclusion

GPU works well, but only for LARGE memory problems. This is because loading small data to memory and using GPU for calculation is overkill, so the CPU has an advantage in this case. But if you have large data dimensions, the GPU can compute  efficiently and surpass the CPU.

This is well known with GPUs: they are only faster if you put a large computational load. It is not specific to pytorch or to MPS...


```python

```
