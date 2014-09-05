"""
like batched_inv, but this implementation runs the sparse matrix stuff in a set of separate processes to speed things up.
"""

import numpy as np
import wmf
import batched_inv

import multiprocessing as mp


class CallableObject(object):
    """
    Hack for multiprocessing stuff
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, arg):
        return self.func(arg, *self.args, **self.kwargs)


def get_rows(S, D, i):
    lo, hi = D.indptr[i], D.indptr[i + 1]
    return S.data[lo:hi], D.data[lo:hi], D.indices[lo:hi]

def build_batch(b, S, D, Y_e, b_y, byY, YTYpR, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f + 1), dtype=dtype)
    B_stack = np.empty((current_batch_size, f + 1, f + 1), dtype=dtype)

    for ib, k in enumerate(xrange(lo, hi)):
        s_u, d_u, i_u = get_rows(S, D, k)

        Y_u = Y_e[i_u] # exploit sparsity
        b_y_u = b_y[i_u]
        A = d_u.dot(Y_u)
        A -= np.dot(b_y_u, (Y_u * s_u[:, None]))
        A -= byY

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B

    return A_stack, B_stack


def recompute_factors_bias_batched_mp(Y, S, D, lambda_reg, dtype='float32', batch_size=1, solve=batched_inv.solve_sequential):
    m = D.shape[0] # m = number of users
    f = Y.shape[1] - 1 # f = number of factors
    
    b_y = Y[:, f] # vector of biases

    Y_e = Y.copy()
    Y_e[:, f] = 1 # factors with added column of ones

    YTY = np.dot(Y_e.T, Y_e) # precompute this

    R = np.eye(f + 1) # regularization matrix
    R[f, f] = 0 # don't regularize the biases!
    R *= lambda_reg

    YTYpR = YTY + R

    byY = np.dot(b_y, Y_e) # precompute this as well

    X_new = np.zeros((m, f + 1), dtype=dtype)

    num_batches = int(np.ceil(m / float(batch_size)))

    rows_gen = wmf.iter_rows(S, D)

    for b in xrange(num_batches):
        A_stack, B_stack = build_batch(b, S, D, Y_e, b_y, byY, YTYpR, batch_size, m, f, dtype)
        X_stack = solve(A_stack, B_stack)
        X_new[lo:hi] = X_stack

    return X_new