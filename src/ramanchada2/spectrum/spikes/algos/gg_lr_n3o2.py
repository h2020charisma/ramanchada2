from typing import Union

import numpy as np
from pydantic import validate_arguments


def metric(y):
    l3o2 = [(11*y[i-1]+2*y[i-2]-7*y[i-3])/6 for i in range(4, len(y)-4)]
    r3o2 = [(11*y[i+1]+2*y[i+2]-7*y[i+3])/6 for i in range(4, len(y)-4)]
    y = y[4:-4]
    metric = np.max([y-np.max([l3o2, r3o2], axis=0), np.min([l3o2, r3o2], axis=0)-y], axis=0)
    return np.pad(metric, (4, 4))


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    return np.where(metric(s) > threshold)[0]
