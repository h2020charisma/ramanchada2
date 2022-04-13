#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from ramanchada2.misc.spectrum_algorithm import spectrum_algorithm_deco


@spectrum_algorithm_deco
def add_poisson_noise(old_spe, new_spe, **kwargs):
    dat = old_spe.y + [np.random.normal(0., np.sqrt(i)) for i in old_spe.y]
    dat[dat < 0] = 0
    new_spe.y = np.array(dat)
