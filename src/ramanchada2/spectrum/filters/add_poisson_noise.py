#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from ramanchada2.misc.spectrum_deco import spectrum_algorithm_deco


@spectrum_algorithm_deco
def add_poisson_noise(old_spe, new_spe, scale=1):
    dat = old_spe.y + [np.random.normal(0., np.sqrt(i*scale)) for i in old_spe.y]
    dat[dat < 0] = 0
    new_spe.y = np.array(dat)
