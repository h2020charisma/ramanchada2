#!/usr/bin/env python3

from typing import Literal, Union, Callable, List

import numpy as np
from numpy.typing import NDArray
from scipy import signal
import lmfit
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def convolve(
        old_spe: Spectrum,
        new_spe: Spectrum, /,
        lineshape: Union[Callable[[Union[float, NDArray]], float],
                         List[float],
                         Literal[
                              'gaussian', 'lorentzian',
                              'voigt', 'pvoigt', 'moffat',
                              ]],
        **kwargs):
    """
    Convole spectrum with a function

    Parameters
    ----------
    lineshape : Union[Callable[[Union[float, NDArray]], float],
                      Literal['gaussian', 'lorentzian', 'voigt', 'pvoigt', 'moffat']]
        predefined model or user defined function to convolve spectrum with
    """

    if isinstance(lineshape, list):
        new_spe.y = signal.convolve(old_spe.y, lineshape, mode='same')
    else:
        if callable(lineshape):
            shape_fun = lineshape
        else:
            shape_fun = getattr(lmfit.lineshapes, lineshape)

        leny = len(old_spe.y)
        x = np.arange(-(leny-1)//2, (leny+1)//2, dtype=float)
        shape_val = shape_fun(x, **kwargs)
        new_spe.y = signal.convolve(old_spe.y, shape_val, mode='same')
