from typing import List

import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_constructor

from ..spectrum import Spectrum


@add_spectrum_constructor(set_applied_processing=True)
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def hdr_from_multi_exposure(spes_in: List[Spectrum]):
    """Create an HDR spectrum from several spectra with different exposures.

    The resulting spectrum will have the details in low-intensity peaks
    from long-exposure-time spectrum. As long-exposure-time
    spectrum might be sturated, the information for high-intensity
    peaks will be taken from short-exposure-time spectrum.
    This function will work on a very limited number of spectra,
    because we still do not have standardized metadata.
    """

    spes = list(sorted(spes_in, key=lambda s: float(s.meta['intigration times(ms)'])))  # type: ignore
    if not np.all([spes[0].x == s.x for s in spes]):
        raise ValueError('x-axes of the spectra should be equal')
    spes_cpms = np.array([s.y / float(s.meta['intigration times(ms)']) for s in spes])  # type: ignore
    masks = np.array(list(map(lambda s: s.y > s.meta['yaxis_max'], spes)))  # type: ignore
    y = spes_cpms[0]
    for si in range(1, len(spes_cpms)):
        y[~masks[si]] = spes_cpms[si][~masks[si]]
    return Spectrum(x=spes[0].x, y=y)
