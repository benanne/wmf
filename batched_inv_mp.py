import numpy as np 
import multiprocessing as mp
import itertools

NUM_PROCESSES = 4

def process_func(tup):
    A, B = tup
    return np.linalg.solve(B.T, A.T).T

def solve_mp(As, Bs):
    pool = mp.Pool(NUM_PROCESSES)
    X_list = pool.map(process_func, itertools.izip(As, Bs))
    pool.close() # IMPORTANT!
    Xs = np.array(X_list)
    return Xs