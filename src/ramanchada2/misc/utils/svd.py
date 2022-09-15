import numpy as np


def svd_solve(A, b):
    """
    Solves Ax=b
    """
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    v = vt.T
    s_1 = 1/s
    s_1[np.abs(s) < 1e-8 * np.max(s)] = 0
    x = v @ (np.diag(s_1) @ (u.T@b))
    return x


def svd_inverse(mat):
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    s_1 = 1/s
    s_1[np.abs(s) < 1e-8 * np.max(s)] = 0
    return vt.T @ np.diag(s_1) @ u.T
