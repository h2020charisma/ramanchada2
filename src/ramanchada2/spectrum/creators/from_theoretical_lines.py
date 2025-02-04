from typing import Dict, List, Literal, Union

import numpy as np
import numpy.typing as npt
from lmfit import lineshapes
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_constructor

from ..spectrum import Spectrum


@validate_call(config=dict(arbitrary_types_allowed=True))
def generate_theoretical_lines(*,
                               shapes: List[Literal[lineshapes.functions]],  # type: ignore
                               params: List[Dict],
                               x: npt.NDArray[np.float64]):
    y = np.zeros_like(x, dtype=float)
    for shape_name, pars in zip(shapes, params):
        shape = getattr(lineshapes, shape_name)
        y += shape(x=x, **pars)
    return y


@add_spectrum_constructor()
@validate_call(config=dict(arbitrary_types_allowed=True))
def from_theoretical_lines(
        shapes: List[Literal[lineshapes.functions]],  # type: ignore
        params: List[Dict],
        x: Union[int, npt.NDArray[np.float64]] = 2000):
    """
    Generate spectrum from `lmfit` shapes.

    Args:
        shapes:
            The shapes to be used for spectrum generation.
        params:
            Shape parameters to be applied to be used with shapes.
        x:
            Array with `x` values, by default `np.array(2000)`.
    """
    spe = Spectrum(x=x)
    spe.y = generate_theoretical_lines(shapes=shapes, params=params, x=spe.x)
    return spe
