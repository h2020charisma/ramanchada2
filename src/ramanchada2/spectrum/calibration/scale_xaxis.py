from typing import Callable, Union

import numpy as np
import numpy.typing as npt
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def scale_xaxis_linear(old_spe: Spectrum,
                       new_spe: Spectrum, /,
                       factor: float = 1,
                       preserve_integral: bool = False):
    r"""
    Scale x-axis using a factor.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        factor: Defaults to 1.
            Multiply x-axis values with `factor`
        preserve_integral: optional. Defaults to False.
            If True, preserves the integral in sence
            $\sum y_{orig;\,i}*{\Delta x_{orig}}_i = \sum y_{new;\,i}*{\Delta x_{new}}_i = $
    Returns: Corrected spectrum
    """
    new_spe.x = old_spe.x * factor
    if preserve_integral:
        new_spe.y = old_spe.y / factor


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def scale_xaxis_fun(old_spe: Spectrum,
                    new_spe: Spectrum, /,
                    fun: Callable[[Union[int, npt.NDArray]], float],
                    args=[]):
    """
    Apply arbitrary calibration function to the x-axis values.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        fun: function to be applied
        args: Additional arguments to the provided functions

    Returns: Corrected spectrum

    Raises:
        ValueError: If the new x-values are not strictly monotonically increasing.
    """
    new_spe.x = fun(old_spe.x, *args)
    if (np.diff(new_spe.x) < 0).any():
        raise ValueError('The provided function is not a monoton increasing funciton.')
