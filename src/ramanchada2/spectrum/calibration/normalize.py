#!/usr/bin/env python3

from typing import Literal
import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def normalize(old_spe: Spectrum,
              new_spe: Spectrum, /,
              strategy: Literal['unity', 'min_unity', 'unity_continuous', 'minmax'] = 'minmax'):
    if strategy == 'unity':
        res = old_spe.y
        res /= np.sum(res)
        new_spe.y = res
    elif strategy == 'min_unity':
        res = old_spe.y - np.min(old_spe.y)
        res /= np.sum(res)
        new_spe.y = res
    if strategy == 'unity_continuous':
        res = old_spe.y
        res /= np.sum(res)
        res /= np.diff(old_spe.x_bin_boundaries)
        new_spe.y = res
    elif strategy == 'minmax':
        res = old_spe.y - np.min(old_spe.y)
        res /= np.max(res)
        new_spe.y = res
