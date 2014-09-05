import numpy as np
import wmf

cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def solve_sequential(As, Bs):
    X_stack = np.empty(As.shape, dtype=As.dtype)

    for k in xrange(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k].T, As[k].T).T

    return X_stack


def solve_sequential_inv(As, Bs):
    X_stack = np.empty(As.shape, dtype=As.dtype)

    for k in xrange(As.shape[0]):
        Binv = np.linalg.inv(Bs[k])
        X_stack[k] = np.dot(As[k], Binv)

    return X_stack


def recompute_factors_bias_batched(np.ndarray[DTYPE_t, ndim=2] Y, S, D, float lambda_reg, dtype='float32', int batch_size=1, solve=solve_sequential):
    """
    Like recompute_factors_bias, but the inversion/solving happens in batches
    and is performed by a solver function that can also be swapped out.
    """
    assert dtype == 'float32'

    cdef int m = D.shape[0] # m = number of users
    cdef int f = Y.shape[1] - 1 # f = number of factors
    
    cdef np.ndarray[DTYPE_t, ndim=1] b_y = Y[:, f] # vector of biases

    cdef np.ndarray[DTYPE_t, ndim=2] Y_e = Y.copy()
    Y_e[:, f] = 1 # factors with added column of ones

    cdef np.ndarray[DTYPE_t, ndim=2] YTY = np.dot(Y_e.T, Y_e) # precompute this

    cdef np.ndarray[DTYPE_t, ndim=2] R = np.eye(f + 1, dtype=dtype) # regularization matrix
    R[f, f] = 0 # don't regularize the biases!
    R *= lambda_reg

    cdef np.ndarray[DTYPE_t, ndim=2] YTYpR = YTY + R

    cdef np.ndarray[DTYPE_t, ndim=1] byY = np.dot(b_y, Y_e) # precompute this as well

    cdef np.ndarray[DTYPE_t, ndim=2] X_new = np.zeros((m, f + 1), dtype=dtype)

    cdef int num_batches = int(np.ceil(m / float(batch_size)))

    rows_gen = wmf.iter_rows(S, D)

    cdef int lo, hi, current_batch_size
    cdef np.ndarray[DTYPE_t, ndim=2] A_stack
    cdef np.ndarray[DTYPE_t, ndim=3] B_stack

    cdef np.ndarray[DTYPE_t, ndim=2] Y_u
    cdef np.ndarray[DTYPE_t, ndim=1] b_y_u

    cdef np.ndarray[DTYPE_t, ndim=1] A
    cdef np.ndarray[DTYPE_t, ndim=2] B
    cdef np.ndarray[DTYPE_t, ndim=2] YTSY

    for b in range(num_batches):
        lo = b * batch_size
        hi = min((b + 1) * batch_size, m)
        current_batch_size = hi - lo

        A_stack = np.empty((current_batch_size, f + 1), dtype=dtype)
        B_stack = np.empty((current_batch_size, f + 1, f + 1), dtype=dtype)

        for ib in range(current_batch_size):
            k, s_u, d_u, i_u = rows_gen.next()

            Y_u = Y_e[i_u] # exploit sparsity
            b_y_u = b_y[i_u]
            A = d_u.dot(Y_u)
            A -= np.dot(b_y_u, (Y_u * s_u[:, None]))
            A -= byY

            YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
            B = YTSY + YTYpR

            A_stack[ib] = A
            B_stack[ib] = B

        print "start batch solve %d" % b
        X_stack = solve(A_stack, B_stack)
        print "finished"
        X_new[lo:hi] = X_stack

    return X_new