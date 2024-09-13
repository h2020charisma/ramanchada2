import numpy as np
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def pad_zeros(old_spe: Spectrum,
              new_spe: Spectrum, /):
    """
    Extend x-axis by 100% in both directions.

    The x-axis of resultant spectrum will be:
    $[x_{lower}-(x_{upper}-x_{lower})..(x_{upper}+(x_{upper}-x_{lower}))]$.
    The length of the new spectrum is 3 times the original. The added values
    are with an uniform step. In the middle is the original spectrum with
    original x and y values. The coresponding y vallues for the newly added
    x-values are always zeros.

    Args:
        old_spe: internal use only
        new_spe: internal use only

    Returns: modified Spectrum
    """
    lenx = len(old_spe.x)
    minx = np.min(old_spe.x)
    maxx = np.max(old_spe.x)
    xl = np.linspace(minx-(maxx-minx), minx, lenx, endpoint=True)[:-1]
    xr = np.linspace(maxx, maxx+(maxx-minx), lenx, endpoint=True)[1:]

    new_spe.y = np.concatenate((np.zeros(lenx-1), old_spe.y, np.zeros(lenx-1)))
    new_spe.x = np.concatenate((xl, old_spe.x, xr))
