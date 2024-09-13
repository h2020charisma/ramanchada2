from typing import Literal, Tuple

import numpy as np
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def trim_axes(old_spe: Spectrum,
              new_spe: Spectrum, /,
              method: Literal['x-axis', 'bins'],
              boundaries: Tuple[float, float],
              ):
    """
    Trim axes of the spectrum.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        method: 'x-axis' or 'bins'
            If 'x-axis' boundaries will be interpreted as x-axis values.
            If 'bins' boundaries will be interpreted as indices.
        boundaries: lower and upper boundary for the trimming.

    Returns: modified Spectrum
    """
    if method == 'bins':
        lb = int(boundaries[0])
        rb = int(boundaries[1])
    elif method == 'x-axis':
        lb = int(np.argmin(np.abs(old_spe.x - boundaries[0])))
        rb = int(np.argmin(np.abs(old_spe.x - boundaries[1])))
    new_spe.x = old_spe.x[lb:rb]
    new_spe.y = old_spe.y[lb:rb]
