import numpy as np
import wmf
import batched_inv
import batched_inv_precompute
import solve_mp
import solve_gpu

np.random.seed(123)

B = np.load("test_matrix_large.pkl")

# shuffle columns of B so the dense parts are evenly distributed
indices = np.arange(B.shape[1])
np.random.shuffle(indices)
B = B[:, indices]

S = wmf.log_surplus_confidence_matrix(B, alpha=2.0, epsilon=1e-6)


num_factors = 40 + 1
num_iterations = 1
batch_size = 10000

# solve = batched_inv.solve_sequential
# solve = solve_mp.solve_mp
solve = solve_gpu.solve_gpu


# U, V = wmf.factorize(S, num_factors=num_factors, lambda_reg=1e-5, num_iterations=num_iterations, init_std=0.01, verbose=True, dtype='float32',
#     recompute_factors=batched_inv.recompute_factors_bias_batched, batch_size=batch_size, solve=solve)


U, V = wmf.factorize(S, num_factors=num_factors, lambda_reg=1e-5, num_iterations=num_iterations, init_std=0.01, verbose=True, dtype='float32',
    recompute_factors=batched_inv_precompute.recompute_factors_bias_batched_precompute, batch_size=batch_size, solve=solve)