from typing import Union

import numpy as np
from pydantic import validate_call

from .double_spike import metric as sp2_metric
from .single_spike import metric as sp1_metric
from .double_spike import bool_hot as sp2_bool_hot
from .single_spike import bool_hot as sp1_bool_hot


@validate_call()
def bool_hot(s, *,
             single_spike_threshold: Union[None, float] = None,
             double_spike_threshold: Union[None, float] = None,
             ):
    base = sp1_bool_hot(s, threshold=single_spike_threshold)
    ext = sp2_bool_hot(s, threshold=double_spike_threshold)
    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return idx


@validate_call()
def indices_(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    base = sp1_metric(s) > threshold
    ext = sp2_metric(s) > threshold
    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return np.where(idx)[0]
