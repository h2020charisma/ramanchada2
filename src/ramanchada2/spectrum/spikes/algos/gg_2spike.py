import numpy as np
from scipy.stats import median_abs_deviation


def metric(s):
    """
    https://doi.org/10.48550/arXiv.2401.09196
    """

    m = [np.abs(s[i+2]-s[i+1]-s[i]+s[i-1]) - np.abs(s[i+2]-s[i+1]+s[i]-s[i-1]) for i in range(1, len(s)-2)]
    m = np.pad(m, [1, 2], mode='edge')
    m /= median_abs_deviation(m)
    return m


def indices(s, threshold=None):
    if threshold is None:
        threshold = 12
    return np.where(metric(s) > threshold)[0]
