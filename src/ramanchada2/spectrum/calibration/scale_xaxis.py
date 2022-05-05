#!/usr/bin/env python3

from typing import Callable

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
    new_spe.x = fun(old_spe.x)
