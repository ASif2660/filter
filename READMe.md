Author: M Asif C

There is a high chance you might come across this error

warning: calling a __host__ function from a __host__ __device__ function is not allowed

But it's not a problem :), it's kind of legal in the language of clang but yeah I don't care if it works it works


Also do remember to upgrade boost and eigen if in case.

I had to use local eigen version instead of the one at /usr/ as I have already built important libraries with the existing one and
I would prefer not tamper it.


#It turns out eigen matrices cannot be utilised directly into GPU as they are optimised for CPU's
#https://stackoverflow.com/questions/41119365/using-eigen-3-3-in-a-cuda-kernel
