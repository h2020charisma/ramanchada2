#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from lmfit import Model
from lmfit import Parameters
from typing import Union
from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import spectrum_constructor_deco
from ramanchada2.misc.types import SpectrumMetaData


@spectrum_constructor_deco
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_theoretical_lines(
        spe: Spectrum, /,
        model: Model,
        params: Parameters,
        x: Union[int, npt.NDArray[np.float64]] = 2000,
        metadata: SpectrumMetaData = {}):
    """
    Generate spectrum from `lmfit` model.

    Parameters
    ----------
    model : lmfit.Model
        the model to be used for spectrum generation
    params : lmfit.Parameters
        the parameters to be applied to the model
    x : Union[int, npt.NDArray[np.float64]], optional
        array with x values, by default np.array(2000)
    metadata : Dict[str, Union[int, str, bool]], optional
        metadata for the newly created `Spectrum`, by default {}
    """
    if isinstance(x, np.ndarray):
        spe.x = x
    else:
        spe.x = np.arange(x)
    spe._ydata = model.eval(params=params, x=spe.x)
    spe.meta = metadata
