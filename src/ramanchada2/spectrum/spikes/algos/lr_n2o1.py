from typing import Union

import numpy as np
from pydantic import validate_arguments

from .lin_reg_extrap import lr_extrap_n2_l1, lr_extrap_n2_r1


def metric(y):
    l2o1 = lr_extrap_n2_l1(y)
    r2o1 = lr_extrap_n2_r1(y)
    metric = np.max([y-np.max([l2o1, r2o1], axis=0),
                     np.min([l2o1, r2o1], axis=0)-y],
                    axis=0)
    metric[:2] = 0
    metric[-2:] = 0
    return metric


@validate_arguments()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    return np.where(metric(s) > threshold)[0]
