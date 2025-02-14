from typing import Union

import numpy as np
from pydantic import validate_call

from .double_spike import bool_hot as double_spike_bool_hot
from .lr_n2o1 import bool_hot as n2o1_bool_hot
from .lr_n2o2 import bool_hot as n2o2_bool_hot
from .lr_n3o1 import bool_hot as n3o1_bool_hot
from .lr_n3o2 import bool_hot as n3o2_bool_hot
from .single_spike import bool_hot as single_spike_bool_hot


@validate_call()
def bool_hot(s, *,
             n2o1_threshold: Union[None, float] = None,
             n2o2_threshold: Union[None, float] = None,
             ):
    n2o1 = n2o1_bool_hot(s)
    n2o2 = n2o2_bool_hot(s)
    n3o1 = n3o1_bool_hot(s)
    n3o2 = n3o2_bool_hot(s)
    sp1 = single_spike_bool_hot(s)
    sp2 = double_spike_bool_hot(s)
    base = n2o1 | n3o1 | sp1
    ext = n2o2 | n3o2 | sp2
    # metrics are padded with zeros so this is safe
    idx = ext & np.roll(base, 1)
    idx |= ext & np.roll(base, -1)
    idx |= base
    return idx
