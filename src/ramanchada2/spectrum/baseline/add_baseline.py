from typing import Union, Callable

from pydantic import validate_call, Field
import numpy as np
from scipy import signal, fft

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@validate_call(config=dict(arbitrary_types_allowed=True))
def generate_baseline(
        n_freq: int = Field(..., gt=2),
        size: int = Field(..., gt=2),
        # validation for rng_seed is removed because
        # it makes in-place modification impossible
        rng_seed=None):
    if isinstance(rng_seed, dict):
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_seed
    else:
        rng = np.random.default_rng(rng_seed)
    k = rng.normal(0, size, size=(2, n_freq))
    k[1][0] = 0
    z = k[0] + k[1]*1j
    w = signal.windows.bohman(2*len(z))[-len(z):]
    z *= w
    z = np.concatenate([z, np.zeros(size-n_freq)])
    base = fft.irfft(z)
    base = base[:size]
    base -= base.min()
    base /= base.max()
    if isinstance(rng_seed, dict):
        rng_seed.update(rng.bit_generator.state)
    return base


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def add_baseline(old_spe: Spectrum, new_spe: Spectrum, /, n_freq: int, amplitude: float, pedestal: float = 0,
                 func: Union[Callable, None] = None, rng_seed=None):
    """
    Add artificial baseline to the spectrum.
    A random baseline is generated in frequency domain using uniform random numbers.
    The baseline in frequency domain is tapered with bohman window to reduce the bandwidth
    of the baseline to first `n_freq` frequencies and is transformed to "time" domain.
    Additionaly by using `func` parameter the user can define arbitrary function
    to be added as baseline.

    Args:
        n_freq:
            Must be `> 2`. Number of lowest frequency bins distinct from zero.
        amplitude:
            Upper boundary for the uniform random generator.
        pedestal:
            Additive constant pedestal to the spectrum.
        func:
            Callable. User-defined function to be added as baseline. Example: `func = lambda x: x*.01 + x**2*.0001`.
        rng_seed:
            `int`, optional. Seed for the random generator.
    """
    size = len(old_spe.y)
    base = generate_baseline(n_freq=n_freq, size=size, rng_seed=rng_seed)
    y = old_spe.y + amplitude*base + pedestal
    if func is not None:
        y += func(old_spe.x) + old_spe.y
    new_spe.y = y
