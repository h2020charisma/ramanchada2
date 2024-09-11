import numpy as np
from pydantic import PositiveFloat, confloat, validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@validate_call(config=dict(arbitrary_types_allowed=True))
def generate_add_gaussian_noise_drift(y, /,
                                      sigma: PositiveFloat,
                                      coef: confloat(ge=0, le=1),  # type: ignore [valid-type]
                                      # validation for rng_seed is removed because
                                      # it makes in-place modification impossible
                                      rng_seed=None):
    if isinstance(rng_seed, dict):
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_seed
    else:
        rng = np.random.default_rng(rng_seed)
    gaus = rng.normal(0., sigma+coef/np.sqrt(2), size=len(y))
    cs = np.cumsum(gaus)
    # coef*sum(cs[:i]) + (1-coef)*gaus is identical to
    # coef*sum(cs[:i-1]) + gaus
    noise = coef*cs + gaus*(1-coef)
    noise -= np.std(noise)
    dat = y + noise
    if any(dat < 0):
        dat += abs(dat.min())
    if isinstance(rng_seed, dict):
        rng_seed.update(rng.bit_generator.state)
    return np.array(dat)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def add_gaussian_noise_drift(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        sigma: PositiveFloat,
        coef: confloat(ge=0, le=1),  # type: ignore [valid-type]
        # validation for rng_seed is removed because
        # it makes in-place modification impossible
        rng_seed=None):
    r"""
    Add cumulative gaussian noise to the spectrum.

    Exponential-moving-average-like gaussian noise is added
    to each sample. The goal is to mimic the low-frequency noise
    (or random substructures in spectra).
    The additive noise is
    .. math::
        a_i = coef*\sum_{j=0}^{i-1}g_j + g_i,
    where
    .. math::
        g_i = \mathcal{N}(0, 1+\frac{coef}{\sqrt 2}).
    This way drifting is possible while keeping the
    .. math::
        \sigma(\Delta(a)) \approx 1.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        sigma:
            Sigma of the gaussian noise.
        coef:
            `float` in `[0, 1]`, drifting coefficient. If `coef == 0`,
            the result is identical to `add_gaussian_noise()`.
        rng_seed:
            `int` or rng state, optional. Seed for the random generator.
            If a state is provided, it is updated in-place.

    Returns: modified Spectrum
    """
    new_spe.y = generate_add_gaussian_noise_drift(old_spe.y,
                                                  sigma=sigma,
                                                  coef=coef,
                                                  rng_seed=rng_seed)
