from typing import Union

import numpy as np
from pydantic import validate_arguments


def metric(y):
    le = [(4*y[i-1] + y[i-2] - 2*y[i-3])/3 for i in range(3, len(y)-3)]
    ri = [(4*y[i+1] + y[i+2] - 2*y[i+3])/3 for i in range(3, len(y)-3)]
    le = np.pad(le, (3, 3), mode='edge')
    ri = np.pad(ri, (3, 3), mode='edge')
    max = np.max([le, ri], axis=0)
    min = np.min([le, ri], axis=0)
    metric = np.max([y-max, min-y], axis=0)
    metric[:2] = 0
    metric[-2:] = 0
    return metric


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    return np.where(metric(s) > threshold)[0]
