#!/usr/bin/env python3

from typing import Union, Dict

import numpy as np
import numpy.typing as npt
from scipy import sparse
from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor


@add_spectrum_constructor()
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_delta_lines(
        deltas: Dict[float, float],
        x: Union[int, npt.NDArray] = 2000,
        **kwargs
        ):
    """
    Generate `Spectrum` with delta lines.

    Parameters
    ----------
    deltas : Dict[float, float]
        keys of the dictionary are the x indexes of the deltas;
        values are the amplitudes of the corresponding deltas
    x : Union[int, npt.NDArray], optional
        array with x values. If an integer is provided, it is used
        generate a sequence with `np.arange()`, by default 2000
    """
    x_loc = list(deltas.keys())
    ampl = list(deltas.values())
    spe = Spectrum(x=x, **kwargs)
    y = sparse.coo_array((ampl, (np.zeros_like(x_loc), x_loc)),
                         shape=(1, len(spe.x)))
    y = y.toarray()
    y.shape = (-1)
    spe.y = y
    return spe
