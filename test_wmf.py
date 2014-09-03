import numpy as np
import wmf


B = np.load("test_matrix.pkl")

P = wmf.binarize_matrix(B)
S = wmf.log_surplus_confidence_matrix(B, alpha=2.0, epsilon=1e-6)

U, V = wmf.factorize(P, S, num_factors=40, lambda_reg=1e-5, num_iterations=20, init_std=0.01, verbose=True, dtype='float32')