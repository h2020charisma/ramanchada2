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
    r"""
    Add poisson noise to the spectrum.
    For each particular sample the noise is proportional to $\sqrt{scale*a_i}$.

    Args:
        scale:
            `float`, optional, default is `1`. Scale the amplitude of the noise.
        rng_seed:
            `int` or rng state, optional. Seed for the random generator. If a state is provided, it is updated
            in-place.
    """
    if isinstance(rng_seed, dict):
        rng = np.random.default_rng()
        rng.__setstate__(rng_seed)
    else:
        rng = np.random.default_rng(rng_seed)
    dat = old_spe.y + [rng.normal(0., np.sqrt(i*scale)) for i in old_spe.y]
    dat[dat < 0] = 0
    if isinstance(rng_seed, dict):
        rng_seed.update(rng.__getstate__())
    new_spe.y = np.array(dat)
