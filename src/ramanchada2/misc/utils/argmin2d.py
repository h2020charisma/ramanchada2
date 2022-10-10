import numpy as np


def argmin2d(A):
    ymin_idx = np.argmin(A, axis=0)
    xmin_idx = np.argmin(A, axis=1)
    x_idx = np.unique(xmin_idx[xmin_idx[ymin_idx[xmin_idx]] == xmin_idx])
    y_idx = np.unique(ymin_idx[ymin_idx[xmin_idx[ymin_idx]] == ymin_idx])
    matches = np.stack([y_idx, x_idx]).T
    return matches
