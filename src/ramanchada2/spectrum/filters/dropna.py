import numpy as np
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def dropna(old_spe: Spectrum,
           new_spe: Spectrum):
    """
    Remove non finite numbers on both axes

    Args:
        old_spe: internal use only
        new_spe: internal use only

    Returns: modified Spectrum
    """

    x = old_spe.x
    y = old_spe.y
    idx = np.isfinite(x)
    x = x[idx]
    y = y[idx]
    idx = np.isfinite(y)
    x = x[idx]
    y = y[idx]
    new_spe.x = x
    new_spe.y = y
