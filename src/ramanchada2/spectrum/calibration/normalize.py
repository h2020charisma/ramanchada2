#!/usr/bin/env python3

import numpy as np

from ramanchada2.misc.spectrum_algorithm import spectrum_algorithm_deco


@spectrum_algorithm_deco
def normalize(old_spe, new_spe, strategy='minmax', **kwargs):
    if strategy == 'unity':
        res = old_spe.y - np.min(old_spe.y)
        res /= np.sum(res)
        new_spe.y = res
    elif strategy == 'minmax':
        res = old_spe.y - np.min(old_spe.y)
        res /= np.max(res)
        new_spe.y = res
