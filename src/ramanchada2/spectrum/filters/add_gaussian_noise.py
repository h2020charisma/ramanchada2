import numpy as np
from pydantic import PositiveFloat, validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def add_gaussian_noise(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        sigma: PositiveFloat,
        # validation for rng_seed is removed because
        # it makes in-place modification impossible
        rng_seed=None):
    r"""
    Add gaussian noise to the spectrum.
    Random number i.i.d. $N(0, \sigma)$ is added to every sample

    Args:
        sigma:
            Sigma of the gaussian noise.
        rng_seed:
            `int` or rng state, optional, seed for the random generator. If a state is provided, it is updated
            in-place.
    """
    if isinstance(rng_seed, dict):
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_seed
    else:
        rng = np.random.default_rng(rng_seed)
    dat = old_spe.y + rng.normal(0., sigma, size=len(old_spe.y))
    if any(dat < 0):
        dat += abs(dat.min())
    if isinstance(rng_seed, dict):
        rng_seed.update(rng.bit_generator.state)
    new_spe.y = np.array(dat)
