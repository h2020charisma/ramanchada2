from typing import Literal

import numpy as np
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def normalize(old_spe: Spectrum,
              new_spe: Spectrum, /,
              strategy: Literal['unity', 'min_unity', 'unity_density', 'unity_area', 'minmax',
                                'L1', 'L2'] = 'minmax'):
    """
    Normalize the spectrum.

    Args:
        strategy:
            If `unity`: normalize to `sum(y)`. If `min_unity`: subtract the minimum and normalize to 'unity'. If
            `unity_density`: normalize to `Σ(y_i*Δx_i)`. If `unity_area`: same as `unity_density`. If `minmax`: scale
            amplitudes in range `[0, 1]`. If 'L1' or 'L2': L1 or L2 norm without subtracting the pedestal.
    """
    # NB: old_spe.y returns the spectrum's internal read-only array; never modify it
    # in place -- always compute a new array (plain division allocates one).
    if strategy == 'unity':
        res = old_spe.y
        new_spe.y = res / np.sum(res)
    elif strategy == 'min_unity':
        res = old_spe.y - np.min(old_spe.y)
        new_spe.y = res / np.sum(res)
    if strategy == 'unity_density' or strategy == 'unity_area':
        res = old_spe.y
        new_spe.y = res / np.sum(res * np.diff(old_spe.x_bin_boundaries))
    elif strategy == 'minmax':
        res = old_spe.y - np.min(old_spe.y)
        new_spe.y = res / np.max(res)
    elif strategy == 'L1':
        res = old_spe.y
        new_spe.y = res / np.linalg.norm(res, 1)
    elif strategy == 'L2':
        res = old_spe.y
        new_spe.y = res / np.linalg.norm(res)
