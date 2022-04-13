#!/usr/bin/env python3

from __future__ import annotations

from ramanchada2.misc.spectrum_algorithm import spectrum_algorithm_deco


@spectrum_algorithm_deco
def scale_xaxis_linear(old_spe, new_spe, factor=1, preserve_integral=False, **kwargs):
    new_spe.x = old_spe.x * factor
    if preserve_integral:
        new_spe.y = old_spe.y / factor
