from typing import Union

import numpy as np
from pydantic import validate_call

from .lr_n2o1 import metric as n2o1_metric
from .lr_n2o2 import metric as n2o2_metric


@validate_call()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    m1 = n2o1_metric(s)
    m2 = n2o2_metric(s)
    i1 = m1 > threshold
    i2 = m2 > threshold
    idx = i1 & i2
    return np.where(idx)[0]
