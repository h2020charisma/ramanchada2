#!/usr/bin/env python3

from typing import Literal, Tuple

import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def trim_axes(old_spe: Spectrum,
              new_spe: Spectrum, /,
              method: Literal['x-axis', 'bins'],
              boundaries: Tuple[float, float],
              ):
    if method == 'bins':
        lb = int(boundaries[0])
        rb = int(boundaries[1])
    elif method == 'x-axis':
        lb = int(np.argmin(np.abs(old_spe.x - boundaries[0])))
        rb = int(np.argmin(np.abs(old_spe.x - boundaries[1])))
    new_spe.x = old_spe.x[lb:rb]
    new_spe.y = old_spe.y[lb:rb]
