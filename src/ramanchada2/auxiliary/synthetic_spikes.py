import logging
from typing import Annotated, Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
from pydantic import Field, NonNegativeInt, PositiveInt, validate_call

from ..spectrum.baseline.add_baseline import generate_baseline
from ..spectrum.creators.from_theoretical_lines import \
    generate_theoretical_lines
from ..spectrum.filters.add_gaussian_noise import generate_add_gaussian_noise
from ..spectrum.spikes.spikes import add_spike

logger = logging.getLogger(__name__)


@validate_call
def synthetic_spectra_array(size: PositiveInt = 100000,
                            filename: Optional[str] = None,
                            spe_len: PositiveInt = 100,
                            mode: Literal['r', 'r+', 'w+', 'c'] = 'r+') -> npt.ArrayLike:
    """
    Create numpy array for spiked synthetic spectra

    Args:
        size (PositiveInt, optional): number of entries in the array. Defaults to 100000.
        filename (Optional[str], optional): If a filename is provided,
            the array will be stored as a file. Defaults to None.
        spe_len (PositiveInt, optional): number of bins per spectrum. Defaults to 100.
        mode (Literal[&#39;r&#39;, &#39;r, optional): mode as in `np.memmap`. Defaults to 'r+'.

    Returns:
        npt.ArrayLike
    """
    spe_type = np.dtype([('spe_raman', 'f4', spe_len),
                         ('spe_raman_baseline', 'f4', spe_len),
                         ('spe_raman_baseline_noise', 'f4', spe_len),
                         ('spe_raman_baseline_noise_spike', 'f4', spe_len),
                         ('raman_pos', 'f4'),
                         ('raman_amp', 'f4'),
                         ('raman_fwhm', 'f4'),
                         ('raman_sigma', 'f4'),
                         ('raman_beta', 'f4'),
                         ('spike_fine_pos', 'f4'),
                         ('spike_coarse_pos', 'u1') if spe_len < 256 else ('spike_coarse_pos', 'u2'),
                         ('spike_amp', 'f4'),
                         ('baseline1_nfreq', 'u1'),
                         ('baseline1_amp', 'f4'),
                         ('baseline2_nfreq', 'u1'),
                         ('baseline2_amp', 'f4'),
                         ('noise_amp', 'f4'),
                         ])
    if filename:
        return np.memmap(filename, dtype=spe_type, mode=mode, shape=size)
    else:
        return np.zeros(size, dtype=spe_type)


def generate_spiked_spectrum(inplace, *,
                             x: npt.NDArray,
                             raman_fwhm_fn: Callable[[], float] = lambda: np.random.uniform(3, 10),
                             raman_beta_fn: Callable[[], float] = lambda: 1/(np.random.uniform(.05, .95)),
                             raman_amp_fn: Callable[[], float] = lambda: np.random.uniform(5, 100),
                             raman_pos_fn: Callable[[], float] = lambda: 49.8,
                             bl1_nfreq_fn: Callable[[], PositiveInt] = lambda: 4,
                             bl1_amp_fn: Callable[[], float] = lambda: 5,
                             bl2_nfreq_fn: Callable[[], PositiveInt] = lambda: 10,
                             bl2_amp_fn: Callable[[], float] = lambda: 3,
                             spike_pos_fn: Callable[[], NonNegativeInt] = lambda: np.random.randint(5, 94),
                             spike_fine_pos_fn: Callable[[], Annotated[float, Field(gt=-.5, lt=.5)]
                                                         ] = lambda: np.random.uniform(-.48, .48),
                             spike_amp_fn: Callable[[], float] = lambda: (np.random.uniform(5, 100)
                                                                          * np.random.choice([-1, 1])),
                             noise_amp_fn: Callable[[], float] = lambda: 1,
                             ):
    """Generate synthetic spectrum and store inplace

    Most arguments are lambda functions which return a sinble number.
    Functions can return randomly generated numbers or constants.

    Args:
        inplace (npt.ArrayLike): single element from an array generated by `synthetic_spectra_array()`.
            Results are stored inplace. For debugging purposes an empty dict can be provided.
        x (npt.NDArray): x-axis array
        raman_fwhm_fn: (Callable[[], float], optional): FWHM of the Raman band.
            Defaults to lambda:np.random.uniform(3, 10).
        raman_beta_fn: (Callable[[], float], optional): Moffat Beta of Raman band.
            Defaults to lambda:1/(np.random.uniform(.05, .95)).
        raman_amp_fn: (Callable[[], float], optional): Amplitude of the Raman band.
            Defaults to lambda:np.random.uniform(5, 100).
        raman_pos_fn: (Callable[[], float], optional): Position of the Raman band.
            Defaults to lambda:49.8.
        bl1_nfreq_fn: (Callable[[], PositiveInt], optional): Number of frequencies for
            a baseline. Defaults to lambda:4.
        bl1_amp_fn: (Callable[[], float], optional): Amplitude of the baseline.
            Defaults to lambda:5.
        bl2_nfreq_fn: (Callable[[], PositiveInt], optional): Number of frequencies
            for an additional baseline. Defaults to lambda:10.
        bl2_amp_fn: (Callable[[], float], optional): Amplitude of the additional baseline.
            Defaults to lambda:3.
        spike_pos_fn: (Callable[[], NonNegativeInt], unsigned int), optional):
            Spike position in number of bins. Defaults to lambda:np.random.randint(5, 94).
        spike_fine_pos_fn: (Callable[[], confloat(gt=-.5, lt=.5)], optional):
            Fine adjustment of the spike. If distinct from 0 the spike amplitude is shared between
            two neighbouring bins proportionally. Defaults to lambda:np.random.uniform(-.48, .48).
        spike_amp_fn: (Callable[[], float], optional): Amplitude of the spike.
            Defaults to lambda:np.random.uniform(5, 100)*np.random.choice([-1, 1]).
        noise_amp_fn: (Callable[[], float], optional): Standard deviation of the added noise.
            If noise_amp is kept equal 1, all amplitudes will be proportional to standard deviation
            of the noise which might be preferred in some cases. Defaults to lambda:1.
    """
    raman_fwhm = raman_fwhm_fn()
    raman_beta = raman_beta_fn()
    raman_pos = raman_pos_fn()
    raman_sigma = raman_fwhm/2/np.sqrt(2**(1/raman_beta)-1)
    raman_amp = raman_amp_fn()
    bl1_nfreq = bl1_nfreq_fn()
    bl1_amp = bl1_amp_fn()
    bl2_nfreq = bl2_nfreq_fn()
    bl2_amp = bl2_amp_fn()
    spike_pos = spike_pos_fn()
    spike_fine_pos = spike_fine_pos_fn()
    spike_amp = spike_amp_fn()
    noise_amp = noise_amp_fn()

    y0 = generate_theoretical_lines(shapes=['moffat'],
                                    params=[dict(beta=raman_beta,
                                                 sigma=raman_sigma,
                                                 center=raman_pos,
                                                 amplitude=raman_amp)
                                            ], x=x)
    bl1 = bl1_amp*generate_baseline(n_freq=bl1_nfreq, size=len(x))
    bl2 = bl2_amp*generate_baseline(n_freq=bl2_nfreq, size=len(x))
    ybl = y0 + bl1 + bl2
    yn = generate_add_gaussian_noise(ybl, sigma=noise_amp)
    ysp = add_spike(x, yn, spike_pos+spike_fine_pos, spike_amp)

    inplace['spe_raman'] = y0
    inplace['spe_raman_baseline'] = ybl
    inplace['spe_raman_baseline_noise'] = yn
    inplace['spe_raman_baseline_noise_spike'] = ysp
    inplace['raman_pos'] = raman_pos
    inplace['raman_amp'] = raman_amp
    inplace['raman_fwhm'] = raman_fwhm
    inplace['raman_sigma'] = raman_sigma
    inplace['raman_beta'] = raman_beta
    inplace['spike_fine_pos'] = spike_fine_pos
    inplace['spike_coarse_pos'] = spike_pos
    inplace['spike_amp'] = spike_amp
    inplace['baseline1_nfreq'] = bl1_nfreq
    inplace['baseline1_amp'] = bl1_amp
    inplace['baseline2_nfreq'] = bl2_nfreq
    inplace['baseline2_amp'] = bl2_amp
    inplace['noise_amp'] = noise_amp
