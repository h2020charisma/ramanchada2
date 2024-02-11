from typing import Union

import numpy as np
from pydantic import validate_arguments


def metric(y):
    l2o1 = [2*y[i-1]-y[i-2] for i in range(2, len(y)-2)]
    r2o1 = [2*y[i+1]-y[i+2] for i in range(2, len(y)-2)]
    y = y[2:-2]
    metric = np.max([y-np.max([l2o1, r2o1], axis=0), np.min([l2o1, r2o1], axis=0)-y], axis=0)
    return np.pad(metric, (2, 2))


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    return np.where(metric(s) > threshold)[0]
