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
    if strategy == 'unity':
        res = old_spe.y
        res /= np.sum(res)
        new_spe.y = res
    elif strategy == 'min_unity':
        res = old_spe.y - np.min(old_spe.y)
        res /= np.sum(res)
        new_spe.y = res
    if strategy == 'unity_density' or strategy == 'unity_area':
        res = old_spe.y
        res /= np.sum(res * np.diff(old_spe.x_bin_boundaries))
        new_spe.y = res
    elif strategy == 'minmax':
        res = old_spe.y - np.min(old_spe.y)
        res /= np.max(res)
        new_spe.y = res
    elif strategy == 'L1':
        res = old_spe.y
        res /= np.linalg.norm(res, 1)
        new_spe.y = res
    elif strategy == 'L2':
        res = old_spe.y
        res /= np.linalg.norm(res)
        new_spe.y = res
