#!/usr/bin/env python3

from typing import Union, Dict

import numpy as np
import numpy.typing as npt
from scipy import sparse
from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import spectrum_constructor_deco
from ramanchada2.misc.types import SpectrumMetaData


@spectrum_constructor_deco
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_delta_lines(
        spe: Spectrum,
        deltas: Dict[float, float],
        x: Union[int, npt.NDArray] = 2000,
        metadata: SpectrumMetaData = {}):
    x_loc = list(deltas.keys())
    ampl = list(deltas.values())
    if isinstance(x, np.ndarray):
        spe.x = x
    else:
        spe.x = np.arange(x)

    y = sparse.coo_array((ampl, (np.zeros_like(x_loc), x_loc)),
                         shape=(1, len(spe.x)))
    y = y.toarray()
    y.shape = (-1)
    spe.y = y

    spe.meta = metadata
