import numpy as np
import wmf


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


def recompute_factors_bias_batched(Y, S, lambda_reg, dtype='float32', batch_size=1, solve=solve_sequential):
    """
    Like recompute_factors_bias, but the inversion/solving happens in batches
    and is performed by a solver function that can also be swapped out.
    """
    m = S.shape[0] # m = number of users
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

    rows_gen = wmf.iter_rows(S)

    for b in xrange(num_batches):
        lo = b * batch_size
        hi = min((b + 1) * batch_size, m)
        current_batch_size = hi - lo

        A_stack = np.empty((current_batch_size, f + 1), dtype=dtype)
        B_stack = np.empty((current_batch_size, f + 1, f + 1), dtype=dtype)

        for ib in xrange(current_batch_size):
            k, s_u, i_u = rows_gen.next()

            Y_u = Y_e[i_u] # exploit sparsity
            b_y_u = b_y[i_u]

            A_stack[ib] = np.dot((1 - b_y_u) * s_u + 1, Y_u) # np.dot(s_u + 1 - (b_y_u * s_u), Y_u)
            B_stack[ib] = np.dot(Y_u.T, (Y_u * s_u[:, None]))

        A_stack -= byY[None, :]
        B_stack += YTYpR[None, :, :]

        print "start batch solve %d" % b
        X_stack = solve(A_stack, B_stack)
        print "finished"
        X_new[lo:hi] = X_stack

    return X_new