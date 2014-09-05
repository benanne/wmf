import numpy as np
import pycuda.gpuarray
import pycuda.autoinit

import scikits.cuda
from scikits.cuda import linalg
from scikits.cuda import cublas

linalg.init()


def bptrs(a):
    """
    Pointer array when input represents a batch of matrices or vectors.

    taken from scikits.cuda tests/test_cublas.py
    """
    
    return pycuda.gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)


allocated_shape = [None]
allocations = [None]

def solve_gpu(As, Bs):
    batch_size, num_factors = As.shape

    if allocated_shape[0] == As.shape: # reuse previous allocations
        As_gpu, Bs_gpu, P_gpu, info_gpu, Cs_gpu, Rs_gpu, A_arr, B_arr, C_arr, R_arr = allocations[0]
        As_gpu.set(As)
        Bs_gpu.set(Bs)
    else: # allocate
        # transfer As and Bs to GPU
        As_gpu = pycuda.gpuarray.to_gpu(As.astype('float32'))
        Bs_gpu = pycuda.gpuarray.to_gpu(Bs.astype('float32'))

        # allocate arrays
        P_gpu = pycuda.gpuarray.empty((batch_size, num_factors), np.int32)
        info_gpu = pycuda.gpuarray.zeros(batch_size, np.int32)
        Cs_gpu = pycuda.gpuarray.empty_like(Bs_gpu) # inverted Bs.
        Rs_gpu = pycuda.gpuarray.empty_like(As_gpu) # final output, As * inverted Bs.
        
        # get pointer arrays
        A_arr = bptrs(As_gpu)
        B_arr = bptrs(Bs_gpu)
        C_arr = bptrs(Cs_gpu)
        R_arr = bptrs(Rs_gpu)

        allocated_shape[0] = As.shape
        allocations[0] = As_gpu, Bs_gpu, P_gpu, info_gpu, Cs_gpu, Rs_gpu, A_arr, B_arr, C_arr, R_arr


    handle = scikits.cuda.misc._global_cublas_handle

    # perform LU factorization
    cublas.cublasSgetrfBatched(handle, num_factors, B_arr.gpudata, num_factors, P_gpu.gpudata, info_gpu.gpudata, batch_size)
    # the LU factorization is now in Bs_gpu!

    # use factorization to perform inversion
    cublas.cublasSgetriBatched(handle, num_factors, B_arr.gpudata, num_factors, P_gpu.gpudata, C_arr.gpudata, num_factors, info_gpu.gpudata, batch_size)
    # the inverted matrices are now in Cs_gpu!

    # compute dot products dot(A, C) = dot(A, Binv). Note that the As are actually vectors!
    transb = 'n'
    transa = 'n'
    N, k, m = Cs_gpu.shape
    N2, l = As_gpu.shape
    n = 1 # As_gpu is a batch of vectors, not matrices, but we treat it as a batch of matrices with leading dimension 1.
    # kind of tricky, but it seems to work. The same goes for the output array Rs_gpu.

    lda = max(1, m)
    ldb = max(1, k)
    ldc = max(1, m)
    alpha = np.float32(1.0)
    beta = np.float32(0.0)

    cublas.cublasSgemmBatched(handle, transb, transa, m, n, k, alpha, C_arr.gpudata,
                lda, A_arr.gpudata, ldb, beta, R_arr.gpudata, ldc, N)

    # the resulting batch of vectors is now in Rs_gpu.
    return Rs_gpu.get()
    