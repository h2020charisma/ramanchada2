from typing import Union

import numpy as np
from pydantic import validate_arguments

from .gg_lr_n2o1_n2o2_mix import metric as n2o1_n2o2_metric
from .gg_lr_n2o2 import metric as n2o2_metric


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    base = n2o1_n2o2_metric(s) > threshold
    ext = n2o2_metric(s) > threshold

    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return np.where(idx)[0]
