#!/usr/bin/env python3

from typing import Callable

import numpy as np

from ramanchada2.misc.spectrum_deco import spectrum_algorithm_deco


@spectrum_algorithm_deco
def scale_xaxis_linear(old_spe, new_spe, factor=1, preserve_integral=False, **kwargs):
    new_spe.x = old_spe.x * factor
    if preserve_integral:
        new_spe.y = old_spe.y / factor


@spectrum_algorithm_deco
def scale_xaxis_fun(old_spe, new_spe,
                    fun: Callable[[int], float],
                    **kwargs):
    x = np.zeros_like(old_spe.x)
    for i in range(len(old_spe.x)):
        x[i] = fun(i)
    new_spe.x = x
