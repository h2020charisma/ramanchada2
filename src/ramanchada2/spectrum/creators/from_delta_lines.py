#!/usr/bin/env python3

from typing import Dict, Callable, Union

import numpy as np
from pydantic import validate_call, PositiveInt

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor


@add_spectrum_constructor()
@validate_call(config=dict(arbitrary_types_allowed=True))
def from_delta_lines(
        deltas: Dict[float, float],
        xcal: Union[Callable[[float], float], None] = None,
        nbins: PositiveInt = 2000,
        **kwargs
        ):
    """
    Generate `Spectrum` with delta lines.

    Args:
        deltas:
            Keys of the dictionary are the `x` positions of the deltas; values are the amplitudes of the corresponding
            deltas.
        xcal:
            Callable, optional. `x` axis calibration function.
        nbins:
            `int`, optional. Number of bins in the spectrum.

    Example:

    This will produce spectrum with 1000 bins in the range `[-1000, 2000)`:
    ```py
    xcal = lambda x: x*3 -1000, nbins=1000
    ```
    """
    if xcal is None:
        dk = list(deltas.keys())
        dkmin, dkmax = np.min(dk), np.max(dk)
        if dkmin == dkmax:
            dkmin, dkmax = dkmin*.8, dkmax*1.2
        else:
            dkmin -= (dkmax-dkmin) * .1
            dkmax += (dkmax-dkmin) * .1
        x = np.linspace(dkmin, dkmax, nbins, endpoint=False, dtype=float)
    else:
        x = np.linspace(xcal(0), xcal(nbins), nbins, endpoint=False)
    y = np.zeros_like(x)
    for pos, ampl in deltas.items():
        idx = np.argmin(np.abs(x - pos))
        y[idx] += ampl
    spe = Spectrum(x=x, y=y, **kwargs)
    return spe
