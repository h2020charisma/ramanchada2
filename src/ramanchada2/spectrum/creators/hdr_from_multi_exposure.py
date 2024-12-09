from typing import List

import numpy as np
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_constructor

from ..spectrum import Spectrum


@add_spectrum_constructor(set_applied_processing=True)
@validate_call(config=dict(arbitrary_types_allowed=True))
def hdr_from_multi_exposure(spes_in: List[Spectrum],
                            meta_exposure_time: str = 'intigration times(ms)',
                            meta_ymax: str = 'yaxis_max'):
    """
    Create an HDR spectrum from several spectra with different exposures.

    The resulting spectrum will have the details in low-intensity peaks
    from long-exposure-time spectrum. As long-exposure-time
    spectrum might be sturated, the information for high-intensity
    peaks will be taken from short-exposure-time spectrum.

    Args:
        spes_in (List[Spectrum]): Set of spectra with different exposures
        meta_exposure_time (str, optional): The name of the metadata parameter
            having the exposure/integration time. The units should be the same
            for all the spectra. Defaults to 'intigration times(ms)'.
        meta_ymax (str, optional): The name fo the metadata parameter having
            the maximum ADC value. This value will be used as a threshold. If
            value in a spectrum is higher, the value will be taken from a
            spectrum with shorter exposure. Defaults to 'yaxis_max'.
    """

    spes = list(sorted(spes_in, key=lambda s: float(s.meta[meta_exposure_time])))  # type: ignore
    if not np.all([spes[0].x == s.x for s in spes]):
        raise ValueError('x-axes of the spectra should be equal')
    spes_cpms = np.array([s.y / float(s.meta[meta_exposure_time]) for s in spes])  # type: ignore
    masks = np.array(list(map(lambda s: s.y > s.meta[meta_ymax], spes)))  # type: ignore
    y = spes_cpms[0]
    for si in range(1, len(spes_cpms)):
        y[~masks[si]] = spes_cpms[si][~masks[si]]
    return Spectrum(x=spes[0].x, y=y)
