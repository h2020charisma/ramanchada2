from typing import Union

import numpy as np
from pydantic import validate_call
from scipy.stats import median_abs_deviation

from .lin_reg_extrap import lr_extrap_n2_l1, lr_extrap_n2_r1


def metric(y):
    y = y/median_abs_deviation(np.diff(y))
    l2o1 = lr_extrap_n2_l1(y)
    r2o1 = lr_extrap_n2_r1(y)
    metric = np.max([y-np.max([l2o1, r2o1], axis=0),
                     np.min([l2o1, r2o1], axis=0)-y],
                    axis=0)
    metric[:2] = 0
    metric[-2:] = 0
    return metric


@validate_call()
def bool_hot(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 6.1
    return metric(s) > threshold


@validate_call()
def indices_(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 6.1
    return np.where(metric(s) > threshold)[0]
