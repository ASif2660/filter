Author: M Asif C

##All the important *.cu and *.cuh are located inside folder mylibrary. 
```
warning: calling a __host__ function from a __host__ __device__ function is not allowed
```
The above error is not a problem :), it's kind of legal in the language of clang but yeah I don't care if it works it works

Note:

#It turns out eigen matrices cannot be utilised directly into GPU as they are optimised for CPU's
#https://stackoverflow.com/questions/41119365/using-eigen-3-3-in-a-cuda-kernel

But however Eigen vectors can be used, if you know how to flatten higher dimensional tensors to a vector ;) it should work out.  

TODO: Fix cuda-support/src * errors.  
TODO: Modular NVCC compilation
