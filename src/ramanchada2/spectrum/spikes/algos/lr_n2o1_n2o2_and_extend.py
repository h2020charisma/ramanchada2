from typing import Union

import numpy as np
from pydantic import validate_call

from .lr_n2o1 import bool_hot as n2o1_bool_hot
from .lr_n2o1 import metric as n2o1_metric
from .lr_n2o2 import bool_hot as n2o2_bool_hot
from .lr_n2o2 import metric as n2o2_metric


@validate_call()
def bool_hot(s, *,
             n2o1_threshold: Union[None, float] = None,
             n2o2_threshold: Union[None, float] = None,
             ):
    i1 = n2o1_bool_hot(s, threshold=n2o1_threshold)
    i2 = n2o2_bool_hot(s, threshold=n2o2_threshold)
    # metrics are padded with zeros so this is safe
    idx = i2 & np.roll(i1, 1)
    idx |= i2 & np.roll(i1, -1)
    idx |= i1 & i2
    return idx


@validate_call()
def indices_(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    m1 = n2o1_metric(s)
    m2 = n2o2_metric(s)
    i1 = m1 > threshold
    i2 = m2 > threshold
    # metrics are padded with zeros so this is safe
    idx = i2 & np.roll(i1, 1)
    idx |= i2 & np.roll(i1, -1)
    idx |= i1 & i2
    return np.where(idx)[0]
