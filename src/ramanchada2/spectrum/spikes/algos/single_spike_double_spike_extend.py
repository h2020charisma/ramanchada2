from typing import Union

import numpy as np
from pydantic import validate_arguments

from .double_spike import metric as sp2_metric
from .single_spike import metric as sp1_metric


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    base = sp1_metric(s) > threshold
    ext = sp2_metric(s) > threshold
    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return np.where(idx)[0]
