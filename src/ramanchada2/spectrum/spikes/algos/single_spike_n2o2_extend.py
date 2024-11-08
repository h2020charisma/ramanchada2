from typing import Union

import numpy as np
from pydantic import validate_call

from .lr_n2o2 import metric as n2o2_metric
from .single_spike import metric as sp1_metric


@validate_call()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    base = sp1_metric(s) > threshold
    ext = n2o2_metric(s) > threshold
    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return np.where(idx)[0]
