from typing import Union

import numpy as np
from pydantic import validate_call

from .lr_n2o1_n2o2_mix import metric as n2o1_n2o2_metric
from .lr_n2o2 import metric as n2o2_metric
from .lr_n2o1_n2o2_mix import bool_hot as n2o1_n2o2_bool_hot
from .lr_n2o2 import bool_hot as n2o2_bool_hot


@validate_call()
def bool_hot(s, *,
             n2o1_n2o2_threshold: Union[None, float] = None,
             n2o2_threshold: Union[None, float] = None,
             ):
    base = n2o1_n2o2_bool_hot(s, threshold=n2o1_n2o2_threshold)
    ext = n2o2_bool_hot(s, threshold=n2o2_threshold)

    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return idx


@validate_call()
def indices_(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    base = n2o1_n2o2_metric(s) > threshold
    ext = n2o2_metric(s) > threshold

    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return np.where(idx)[0]
