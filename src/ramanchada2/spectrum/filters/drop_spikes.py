#!/usr/bin/env python3

import numpy as np
from pydantic import validate_arguments, PositiveFloat

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def drop_spikes(old_spe: Spectrum,
                new_spe: Spectrum, /,
                n_sigma: PositiveFloat = 3):
    """
    Removes single-bin spikes.

    Parameters
    ----------
    n_sigma : float, optional
        by default 3
    """
    yi = old_spe.y[1:-1]
    yi_1 = old_spe.y[:-2]
    yi1 = old_spe.y[2:]
    y_merit = np.abs(2*yi-yi_1-yi1) - np.abs(yi1-yi_1)
    use_idx = y_merit < n_sigma * y_merit.std()
    use_idx = np.concatenate(([True], use_idx, [True]))
    new_spe.x = old_spe.x[use_idx]
    new_spe.y = old_spe.y[use_idx]


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_spikes(old_spe: Spectrum,
               new_spe: Spectrum, /,
               n_sigma: PositiveFloat = 3):
    """
    Get single-bin spikes only.

    Parameters
    ----------
    n_sigma : float, optional
        by default 3
    """
    yi = old_spe.y[1:-1]
    yi_1 = old_spe.y[:-2]
    yi1 = old_spe.y[2:]
    y_merit = np.abs(2*yi-yi_1-yi1) - np.abs(yi1-yi_1)
    use_idx = y_merit < n_sigma * y_merit.std()
    use_idx = np.concatenate(([True], use_idx, [True]))
    new_spe.x = old_spe.x[~use_idx]
    new_spe.y = old_spe.y[~use_idx]
