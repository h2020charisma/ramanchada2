from typing import Union

import numpy as np
from pydantic import validate_arguments


def metric(y):
    #l2o2 = [3*y[i-2]-2*y[i-3] for i in range(3, len(y)-3)]
    #r2o2 = [3*y[i+2]-2*y[i+3] for i in range(3, len(y)-3)]
    l2o2 = 3*y[1: -5] - 2*y[0:-6]
    r2o2 = 3*y[5: -1] - 2*y[6:]
    y = y[3:-3]
    metric = np.max([y-np.max([l2o2, r2o2], axis=0), np.min([l2o2, r2o2], axis=0)-y], axis=0)
    return np.pad(metric, (3, 3))


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    return np.where(metric(s) > threshold)[0]
