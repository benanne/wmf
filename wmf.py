import numpy as np
import time
import itertools



def linear_surplus_confidence_matrix(B, alpha):
    # To construct the surplus confidence matrix, we need to operate only on the nonzero elements.
    # This is not possible: S = alpha * B
    S = B.copy()
    S.data = alpha * S.data
    return S


def log_surplus_confidence_matrix(B, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on the nonzero elements.
    # This is not possible: S = alpha * np.log(1 + B / epsilon)
    S = B.copy()
    S.data = alpha * np.log(1 + S.data / epsilon)
    return S



def iter_rows(S):
    """
    Helper function to iterate quickly over the data and indices of the
    rows of the S matrix. A naive implementation using indexing
    on S is much, much slower.
    """
    for i in xrange(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        yield i, S.data[lo:hi], S.indices[lo:hi]


def recompute_factors(Y, S, lambda_reg, dtype='float32'):
    """
    recompute matrix X from Y.
    X = recompute_factors(Y, S, lambda_reg)
    This can also be used for the reverse operation as follows:
    Y = recompute_factors(X, ST, lambda_reg)
    
    The comments are in terms of X being the users and Y being the items.
    """
    m = S.shape[0] # m = number of users
    f = Y.shape[1] # f = number of factors
    YTY = np.dot(Y.T, Y) # precompute this
    YTYpI = YTY + lambda_reg * np.eye(f)
    X_new = np.zeros((m, f), dtype=dtype)

    for k, s_u, i_u in iter_rows(S):
        Y_u = Y[i_u] # exploit sparsity
        A = np.dot(s_u + 1, Y_u)
        YTSY = np.dot(Y_u.T, (Y_u * s_u.reshape(-1, 1)))
        B = YTSY + YTYpI

        # Binv = np.linalg.inv(B)
        # X_new[k] = np.dot(A, Binv) 
        X_new[k] = np.linalg.solve(B.T, A.T).T # doesn't seem to make much of a difference in terms of speed, but w/e

    return X_new



def recompute_factors_bias(Y, S, lambda_reg, dtype='float32'):
    """
    Like recompute_factors, but the last column of X and Y is treated as
    a bias vector.
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

    for k, s_u, i_u in iter_rows(S):
        Y_u = Y_e[i_u] # exploit sparsity
        b_y_u = b_y[i_u]
        A = np.dot((1 - b_y_u) * s_u + 1, Y_u)
        A -= byY

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        # Binv = np.linalg.inv(B)
        # X_new[k] = np.dot(A, Binv) 
        X_new[k] = np.linalg.solve(B.T, A.T).T # doesn't seem to make much of a difference in terms of speed, but w/e

    return X_new



def factorize(S, num_factors, lambda_reg=1e-5, num_iterations=20, init_std=0.01, verbose=False, dtype='float32', recompute_factors=recompute_factors, *args, **kwargs):
    """
    factorize a given sparse matrix using the Weighted Matrix Factorization algorithm by
    Hu, Koren and Volinsky.

    S: 'surplus' confidence matrix, i.e. C - I where C is the matrix with confidence weights.
        S is sparse while C is not (and the sparsity pattern of S is the same as that of
        the preference matrix, so it doesn't need to be specified separately).

    num_factors: the number of factors.

    lambda_reg: the value of the regularization constant.

    num_iterations: the number of iterations to run the algorithm for. Each iteration consists
        of two steps, one to recompute U given V, and one to recompute V given U.

    init_std: the standard deviation of the Gaussian with which V is initialized.

    verbose: print a bunch of stuff during training, including timing information.

    dtype: the dtype of the resulting factor matrices. Using single precision is recommended,
        it speeds things up a bit.

    recompute_factors: helper function that implements the inner loop.

    returns:
        U, V: factor matrices. If bias=True, the last columns of the matrices contain the biases.
    """
    num_users, num_items = S.shape

    if verbose:
        print "precompute transpose"
        start_time = time.time()

    ST = S.T.tocsr()

    if verbose:
        print "  took %.3f seconds" % (time.time() - start_time)
        print "run ALS algorithm"
        start_time = time.time()

    U = None # no need to initialize U, it will be overwritten anyway
    V = np.random.randn(num_items, num_factors).astype(dtype) * init_std

    for i in xrange(num_iterations):
        if verbose:
            print "  iteration %d" % i
            print "    recompute user factors U"

        U = recompute_factors(V, S, lambda_reg, dtype, *args, **kwargs)

        if verbose:
            print "    time since start: %.3f seconds" % (time.time() - start_time)
            print "    recompute item factors V"

        V = recompute_factors(U, ST, lambda_reg, dtype, *args, **kwargs)

        if verbose:
            print "    time since start: %.3f seconds" % (time.time() - start_time)

    return U, V
