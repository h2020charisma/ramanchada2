from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def __add__(
        old_spe: Spectrum,
        new_spe: Spectrum,
        arg: Union[Spectrum, NDArray, float]):
    if isinstance(arg, Spectrum):
        if not (old_spe.x == arg.x).all():
            ValueError('x axes should be equal')
        new_spe.y = old_spe.y + arg.y
    elif isinstance(arg, np.ndarray):
        if old_spe.y.shape != arg.shape:
            ValueError(f'shapes does not match {old_spe.y.shape} != {arg.shape}')
        new_spe.y = old_spe.y + arg
    elif isinstance(arg, float):
        new_spe.y = old_spe.y + arg
    else:
        ValueError('This should never happen')
