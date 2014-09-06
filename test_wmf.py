import numpy as np
import wmf


B = np.load("test_matrix.pkl")

S = wmf.log_surplus_confidence_matrix(B, alpha=2.0, epsilon=1e-6)

U, V = wmf.factorize(S, num_factors=41, lambda_reg=1e-5, num_iterations=2, init_std=0.01, verbose=True, dtype='float32', recompute_factors=wmf.recompute_factors_bias)