import numpy as np
import wmf
import batched_inv
import batched_inv_mp
import batched_inv_gpu

B = np.load("test_matrix.pkl")

P = wmf.binarize_matrix(B)
S = wmf.log_surplus_confidence_matrix(B, alpha=2.0, epsilon=1e-6)


num_factors = 40 + 1
num_iterations = 1
batch_size = 1000

# solve = batched_inv.solve_sequential
# solve = batched_inv_mp.solve_mp
solve = batched_inv_gpu.solve_gpu


U, V = wmf.factorize(P, S, num_factors=num_factors, lambda_reg=1e-5, num_iterations=num_iterations, init_std=0.01, verbose=True, dtype='float32',
    recompute_factors=batched_inv.recompute_factors_bias_batched, batch_size=batch_size, solve=solve)