from typing import Union

import numpy as np
from pydantic import validate_call
from scipy.stats import median_abs_deviation


def metric(s):
    """
    https://doi.org/10.48550/arXiv.2401.09196
    """

    s = s/median_abs_deviation(np.diff(s))
    m = np.abs(-s[:-2]-s[2:]+2*s[1:-1]) - np.abs(s[2:]-s[:-2])
    m = np.pad(m, [1, 1], mode='edge')
    m[:2] = 0
    m[-2:] = 0
    return m


def metric_(s):
    """
    https://doi.org/10.48550/arXiv.2401.09196
    """

    s = s/median_abs_deviation(np.diff(s))
    m = [np.abs(-s[i-1]-s[i+1]+2*s[i]) - np.abs(s[i+1]-s[i-1])
         for i in range(1, len(s)-1)]
    m = np.pad(m, [1, 1], mode='edge')
    m[:2] = 0
    m[-2:] = 0
    return m


@validate_call()
def bool_hot(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 7.95
    return metric(s) > threshold


@validate_call()
def indices_(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 7.95
    return np.where(metric(s) > threshold)[0]
