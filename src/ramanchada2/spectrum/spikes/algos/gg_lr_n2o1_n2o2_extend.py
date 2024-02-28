from typing import Union

import numpy as np
from pydantic import validate_arguments

from .gg_lr_n2o1 import metric as n2o1_metric
from .gg_lr_n2o2 import metric as n2o2_metric


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    m1 = n2o1_metric(s)
    m2 = n2o2_metric(s)
    i1 = m1 > threshold
    i2 = m2 > threshold
    # metrics are padded with zeros so this is safe
    idx = i2 & np.roll(i1, 1)
    idx |= i2 & np.roll(i1, -1)
    idx |= i1
    return np.where(idx)[0]
