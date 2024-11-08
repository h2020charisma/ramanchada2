from typing import Union

import numpy as np
from pydantic import validate_call

from .lin_reg_extrap import lr_extrap_n3_l2, lr_extrap_n3_r2


def metric(y):
    l3o2 = lr_extrap_n3_l2(y)
    r3o2 = lr_extrap_n3_r2(y)
    metric = np.max([y-np.max([l3o2, r3o2], axis=0),
                     np.min([l3o2, r3o2], axis=0)-y],
                    axis=0)
    metric[:4] = 0
    metric[-4:] = 0
    return metric


@validate_call()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    return np.where(metric(s) > threshold)[0]
