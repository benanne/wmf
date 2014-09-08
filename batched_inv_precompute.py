import numpy as np 
import wmf
import batched_inv


def recompute_factors_bias_batched_precompute(Y, S, lambda_reg, dtype='float32', batch_size=1, solve=batched_inv.solve_sequential):
    """
    Like recompute_factors_bias_batched, but doing a bunch of batch stuff outside the for loop.
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

        lo_batch = S.indptr[lo]
        hi_batch = S.indptr[hi] # hi - 1 + 1

        i_batch = S.indices[lo_batch:hi_batch]
        s_batch = S.data[lo_batch:hi_batch]
        Y_e_batch = Y_e[i_batch]
        b_y_batch = b_y[i_batch]

        # precompute the left hand side of the dot product for computing A for the entire batch.
        a_lhs_batch = (1 - b_y_batch) * s_batch + 1

        # also precompute the right hand side of the dot product for computing B for the entire batch.
        b_rhs_batch = Y_e_batch * s_batch[:, None]

        A_stack = np.empty((current_batch_size, f + 1), dtype=dtype)
        B_stack = np.empty((current_batch_size, f + 1, f + 1), dtype=dtype)

        for k in xrange(lo, hi):
            ib = k - lo # index inside the batch

            lo_iter = S.indptr[k] - lo_batch
            hi_iter = S.indptr[k + 1] - lo_batch

            s_u = s_batch[lo_iter:hi_iter]
            Y_u = Y_e_batch[lo_iter:hi_iter]
            a_lhs_u = a_lhs_batch[lo_iter:hi_iter]
            b_rhs_u = b_rhs_batch[lo_iter:hi_iter]

            A_stack[ib] = np.dot(a_lhs_u, Y_u)
            B_stack[ib] = np.dot(Y_u.T, b_rhs_u)

        A_stack -= byY[None, :]
        B_stack += YTYpR[None, :, :]

        print "start batch solve %d" % b
        X_stack = solve(A_stack, B_stack)
        print "finished"
        X_new[lo:hi] = X_stack

    return X_new