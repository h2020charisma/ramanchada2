#!/usr/bin/env python3

from typing import Union

from pydantic import validate_arguments, Field
import numpy as np
from scipy import signal, fft

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def generate_baseline(
        bandwidth: int = Field(..., gt=2),
        size: int = Field(..., gt=2),
        rng_seed: Union[int, None] = None):
    rng = np.random.default_rng(rng_seed)
    k = rng.normal(0, size, size=(2, bandwidth))
    k[1][0] = 0
    z = k[0] + k[1]*1j
    w = signal.windows.bohman(2*len(z))[-len(z):]
    z *= w
    z = np.concatenate([z, np.zeros(size-bandwidth)])
    base = fft.irfft(z)
    base = base[:size]
    base -= base.min()
    base /= base.max()
    return base


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def add_baseline(old_spe: Spectrum, new_spe: Spectrum, /,
                 bandwidth, amplitude, pedestal, rng_seed=None):
    """
    Add artificial baseline to the spectrum.
    A random spectrum is generated in frequency domain using uniform random numbers.
    The signal in frequency domain is tapered with bohman window to reduce the bandwidth
    and is transformed to "time" domain.

    Parameters
    ----------
    bandwidth : int > 2
        number of lowest frequency bins distinct from zero
    amplitude : float
        upper boundary for the uniform random generator
    pedestal : float
        additive constant pedestal to the spectrum
    rng_seed : int, optional
        seed for the random generator
    """
    size = len(old_spe.y)
    base = generate_baseline(bandwidth=bandwidth, size=size, rng_seed=rng_seed)
    new_spe.y = old_spe.y + amplitude*base + pedestal
