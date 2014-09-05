wmf
===

Weighted matrix factorization in Python

This is an implementation of the weighted matrix factorization algorithm using alternating least squares proposed by Hu, Koren and Volinsky in their 2008 paper "Collaborative filtering for implicit feedback datasets". It uses numpy and scipy.sparse.

A version that performs the numerous matrix inversions needed for the ALS steps in batches is also provided, as well as a GPU implementation of the batch matrix inversion step using scikits.cuda. Currently this requires the latest version of scikits.cuda from git (needs cublasSgetriBatched).

The sparse matrix used in the demo code can be downloaded from here: https://dl.dropboxusercontent.com/u/19706734/test_matrix.pkl