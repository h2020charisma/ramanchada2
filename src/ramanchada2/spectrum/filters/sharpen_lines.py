from typing import List

import emd
import numpy as np
from pydantic import PositiveInt, confloat, validate_call
from scipy import fft, signal

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def derivative_sharpening(old_spe: Spectrum,
                          new_spe: Spectrum, /,
                          filter_fraction: confloat(gt=0, le=1) = .6,  # type: ignore
                          sig_width: confloat(ge=0) = .25,  # type: ignore
                          der2_factor: float = 1,
                          der4_factor: float = .1
                          ):
    """
    Derivative-based sharpening.

    Sharpen the spectrum subtracting second derivative and add fourth derivative.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        filter_fraction `float` in (0; 1]: Default is 0.6
            Depth of filtration
        signal_width: The width of features to be enhanced in sample count
        der2_factor: Second derivative scaling factor
        der4_factor: Fourth derivative scaling factor

    Returns: modified Spectrum
    """
    leny = len(old_spe.y)
    Y = fft.rfft(old_spe.y, n=leny)
    h = signal.windows.hann(int(len(Y)*filter_fraction))
    h = h[(len(h))//2-1:]
    h = np.concatenate((h, np.zeros(len(Y)-len(h))))
    der = np.arange(len(Y))
    der = 1j*np.pi*der/der[-1]
    Y *= h
    Y2 = Y*der**2
    Y4 = Y2*der**2
    y0 = fft.irfft(Y, n=leny)
    y2 = fft.irfft(Y2, n=leny)
    y4 = fft.irfft(Y4, n=leny)
    new_spe.y = y0 - y2/sig_width**2*der2_factor + y4/sig_width**4*der4_factor


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def hht_sharpening(old_spe: Spectrum,
                   new_spe: Spectrum, /,
                   movmin=100
                   ):
    """
    Hilbert-Huang based sharpening.

    In order to reduce the overshooting, moving minimum is subtracted from the result

    Args:
        old_spe: internal use only
        new_spe: internal use only
        movmin: optional. Default is 100
            Window size for moving minimum

    Returns: modified Spectrum
    """
    imfs = emd.sift.sift(old_spe.y).T
    freq_list = list()
    for ansig in signal.hilbert(imfs):
        freq_list.append(emd.spectra.freq_from_phase(
            emd.spectra.phase_from_complex_signal(ansig, ret_phase='unwrapped'), 1))
    freq = np.array(freq_list)
    freq[freq < 0] = 0
    freq[np.isnan(freq)] = 0
    imfsall = imfs.copy()
    imfsall[np.isnan(imfsall)] = 0
    imfsall[freq > .3] = 0
    imfsall *= freq**.5
    ynew = np.sum(imfsall, axis=0)
    new_spe.y = ynew
    new_spe.y = new_spe.subtract_moving_minimum(movmin).normalize().y  # type: ignore
    new_spe.y = new_spe.y * old_spe.y.max() + old_spe.y.min()


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def hht_sharpening_chain(old_spe: Spectrum,
                         new_spe: Spectrum, /,
                         movmin: List[PositiveInt] = [150, 50]
                         ):
    """
    Hilbert-Huang based chain sharpening.

    Sequence of Hilbert-Huang sharpening procedures are performed.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        movmin: List[int], optional. Default is [150, 50]
            The numer of values in the list defines how many iterations
            of HHT_sharpening will be performed and the values define
            the moving minimum window sizes for the corresponding operations.

    Returns: modified Spectrum
    """
    spe = old_spe
    for mm in movmin:
        spe = spe.hht_sharpening(movmin=mm)  # type: ignore
    new_spe.y = spe.y
