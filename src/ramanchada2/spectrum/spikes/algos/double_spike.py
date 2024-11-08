from typing import Union

import numpy as np
from pydantic import validate_call


def metric(s):
    """
    https://doi.org/10.48550/arXiv.2401.09196
    """

    m = np.abs(s[3:]-s[2:-1]-s[1:-2]+s[:-3]) - np.abs(s[3:]-s[2:-1]+s[1:-2]-s[:-3])
    m = np.pad(m, [1, 2], mode='edge')
    m[:2] = 0
    m[-2:] = 0
    return m


def metric_(s):
    """
    https://doi.org/10.48550/arXiv.2401.09196
    """

    m = [np.abs(s[i+2]-s[i+1]-s[i]+s[i-1]) - np.abs(s[i+2]-s[i+1]+s[i]-s[i-1]) for i in range(1, len(s)-2)]
    m = np.pad(m, [1, 2], mode='edge')
    m[:2] = 0
    m[-2:] = 0
    return m


@validate_call()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 12
    left_idx = np.where(metric(s) > threshold)[0]
    right_idx = left_idx+1
    return np.unique([left_idx, right_idx])
