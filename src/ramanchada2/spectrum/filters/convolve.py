#!/usr/bin/env python3

from typing import Literal, Union, Callable

import numpy as np
import numpy.typing as npt
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
                         npt.NDArray,
                         Literal[
                              'gaussian', 'lorentzian',
                              'voigt', 'pvoigt', 'moffat',
                              'pearson4', 'pearson7'
                              ]],
        **kwargs):
    """
    Convole spectrum with arbitrary lineshape.

    Args:
        lineshape:
            Can be a callable, `str` or `np.ndarray`. If callable: should have a single positional argument `x`, e.g.
            `lambda x: np.exp((x/5)**2)`. If predefined peak profile: can be `gaussian`, `lorentzian`, `voigt`,
            `pvoigt`, `moffat` or `pearson4`. If `np.ndarray`: lineshape in samples.
        **kwargs:
            Additional kwargs will be passed to lineshape function.
    """

    if isinstance(lineshape, np.ndarray):
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
