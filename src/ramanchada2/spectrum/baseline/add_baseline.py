#!/usr/bin/env python3

from typing import Union

from pydantic import validate_arguments, Field
import numpy as np
from scipy import signal, fft

from ramanchada2.misc.spectrum_deco import spectrum_algorithm_deco


@validate_arguments
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


@spectrum_algorithm_deco
def add_baseline(old_spe, new_spe, bandwidth, amplitude, pedestal, rng_seed=None):
    size = len(old_spe.y)
    base = generate_baseline(bandwidth=bandwidth, size=size, rng_seed=rng_seed)
    new_spe.y = old_spe.y + amplitude*base + pedestal
