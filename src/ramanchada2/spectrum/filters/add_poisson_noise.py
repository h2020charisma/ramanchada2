import numpy as np
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def add_poisson_noise(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        scale: float = 1,
        # validation for rng_seed is removed because
        # it makes in-place modification impossible
        rng_seed=None):
    r"""
    Add poisson noise to the spectrum.

    For each particular sample the noise is proportional to $\sqrt{scale*a_i}$.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        scale:
            `float`, optional, default is `1`. Scale the amplitude of the noise.
        rng_seed:
            `int` or rng state, optional. Seed for the random generator.
            If a state is provided, it is updated in-place.

    Returns: modified Spectrum
    """
    if isinstance(rng_seed, dict):
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_seed
    else:
        rng = np.random.default_rng(rng_seed)
    dat = old_spe.y + [rng.normal(0., np.sqrt(i*scale)) for i in old_spe.y]
    dat[dat < 0] = 0
    if isinstance(rng_seed, dict):
        rng_seed.update(rng.bit_generator.state)
    new_spe.y = np.array(dat)
