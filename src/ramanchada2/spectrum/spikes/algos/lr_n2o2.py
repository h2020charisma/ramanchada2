from typing import Union

import numpy as np
from pydantic import validate_call
from scipy.stats import median_abs_deviation

from .lin_reg_extrap import lr_extrap_n2_l2, lr_extrap_n2_r2


def metric(y):
    y = y/median_abs_deviation(np.diff(y))
    l2o2 = lr_extrap_n2_l2(y)
    r2o2 = lr_extrap_n2_r2(y)
    metric = np.max([y-np.max([l2o2, r2o2], axis=0),
                     np.min([l2o2, r2o2], axis=0)-y],
                    axis=0)
    metric[:3] = 0
    metric[-3:] = 0
    return metric


@validate_call()
def bool_hot(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 8.35
    return metric(s) > threshold


@validate_call()
def indices_(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 8.35
    return np.where(metric(s) > threshold)[0]
