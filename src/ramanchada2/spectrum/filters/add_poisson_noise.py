#!/usr/bin/env python3

from typing import Union

import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def add_poisson_noise(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        scale: float = 1,
        rng_seed: Union[int, None] = None):
    """
    Add poisson noise to the spectrum.

    Parameters
    ----------
    scale : float, optional
        scale the nose amplitude, by default 1
    rng_seed : Union[int, None], optional
        seed for the random generator, by default None
    """
    rng = np.random.default_rng(rng_seed)
    dat = old_spe.y + [rng.normal(0., np.sqrt(i*scale)) for i in old_spe.y]
    dat[dat < 0] = 0
    new_spe.y = np.array(dat)
