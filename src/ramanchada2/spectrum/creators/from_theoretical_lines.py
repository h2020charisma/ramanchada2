#!/usr/bin/env python3

from __future__ import annotations

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
        spe: Spectrum,
        model: Model,
        params: Parameters,
        x: Union[int, npt.NDArray[np.float64]] = 2000,
        metadata: SpectrumMetaData = {}):
    """
    _summ:ary_

    Parameters
    ----------
    spe : Spectrum
        _description_
    model : Model
        _description_
    params : Parameters
        _description_
    x : Union[int, npt.NDArray[np.float64]], optional
        _description_, by default np.array(2000)
    metadata : Dict[str, Union[int, str, bool]], optional
        _description_, by default {}
    """
    if isinstance(x, np.ndarray):
        spe.x = x
    else:
        spe.x = np.arange(x)
    spe._ydata = model.eval(params=params, x=spe.x)
    spe.meta = metadata
