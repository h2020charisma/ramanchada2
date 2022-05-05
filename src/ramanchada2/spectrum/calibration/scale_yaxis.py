#!/usr/bin/env python3

from __future__ import annotations

from ramanchada2.misc.spectrum_deco import spectrum_algorithm_deco


@spectrum_algorithm_deco
def scale_yaxis_linear(old_spe, new_spe, factor=1, **kwargs):
    new_spe.y = old_spe.y * factor
