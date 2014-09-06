"""
like batched_inv, but this implementation runs the sparse matrix stuff in a set of separate processes to speed things up.
"""

import numpy as np
import wmf
import batched_inv

import multiprocessing as mp
import Queue


def buffered_gen_mp(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer = mp.Queue(maxsize=buffer_size)

    def _buffered_generation_process(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in generating more data
            # when the buffer is full, it only causes extra memory usage and effectively
            # increases the buffer size by one.
            while buffer.full():
                # print "DEBUG: buffer is full, waiting to generate more data."
                time.sleep(sleep_time)

            try:
                data = source_gen.next()
            except StopIteration:
                # print "DEBUG: OUT OF DATA, CLOSING BUFFER"
                buffer.close() # signal that we're done putting data in the buffer
                break

            buffer.put(data)
    
    process = mp.Process(target=_buffered_generation_process, args=(source_gen, buffer))
    process.start()
    
    while True:
        try:
            # yield buffer.get()
            # just blocking on buffer.get() here creates a problem: when get() is called and the buffer
            # is empty, this blocks. Subsequently closing the buffer does NOT stop this block.
            # so the only solution is to periodically time out and try again. That way we'll pick up
            # on the 'close' signal.
            try:
                yield buffer.get(True, timeout=sleep_time)
            except Queue.Empty:
                if not process.is_alive():
                    break # no more data is going to come. This is a workaround because the buffer.close() signal does not seem to be reliable.

                # print "DEBUG: queue is empty, waiting..."
                pass # ignore this, just try again.

        except IOError: # if the buffer has been closed, calling get() on it will raise IOError.
            # this means that we're done iterating.
            # print "DEBUG: buffer closed, stopping."
            break


class CallableObject(object):
    """
    Hack for multiprocessing stuff. This creates a callable wrapper object
    with a single argument, that calls the original function with this argument
    plus any other arguments passed at creation time.
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, arg):
        return self.func(arg, *self.args, **self.kwargs)


def get_row(S, i):
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return S.data[lo:hi], S.indices[lo:hi]


def build_batch(b, S, Y_e, b_y, byY, YTYpR, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f + 1), dtype=dtype)
    B_stack = np.empty((current_batch_size, f + 1, f + 1), dtype=dtype)

    for ib, k in enumerate(xrange(lo, hi)):
        s_u, i_u = get_row(S, k)

        Y_u = Y_e[i_u] # exploit sparsity
        b_y_u = b_y[i_u]
        A = (s_u + 1).dot(Y_u)
        A -= np.dot(b_y_u, (Y_u * s_u[:, None]))
        A -= byY

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B

    return A_stack, B_stack


def recompute_factors_bias_batched_mp(Y, S, lambda_reg, dtype='float32', batch_size=1, solve=batched_inv.solve_sequential, num_batch_build_processes=4):
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

    func = CallableObject(build_batch, S, Y_e, b_y, byY, YTYpR, batch_size, m, f, dtype)

    pool = mp.Pool(num_batch_build_processes)
    batch_gen = pool.imap(func, xrange(num_batches))
    batch_gen_buffered = buffered_gen_mp(batch_gen, buffer_size=2, sleep_time=0.001)

    for b, (A_stack, B_stack) in enumerate(batch_gen):
        lo = b * batch_size
        hi = min((b + 1) * batch_size, m)

        X_stack = solve(A_stack, B_stack)
        X_new[lo:hi] = X_stack

    return X_new