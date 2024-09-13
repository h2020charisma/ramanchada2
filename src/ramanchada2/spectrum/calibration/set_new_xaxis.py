import numpy.typing as npt
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def set_new_xaxis(old_spe: Spectrum,
                  new_spe: Spectrum, /,
                  xaxis: npt.NDArray):
    """
    Substitute x-axis values with new ones

    Args:
        old_spe: internal use only
        new_spe: internal use only
        xaxis: new x-axis values

    Returns: corrected spectrum

    Raises:
        ValueError: If the provided array does not match the shape of the spectrum.
    """
    if old_spe.x.shape != xaxis.shape:
        raise ValueError('Shape of xaxis should match the shape of xaxis of the spectrum')
    new_spe.x = xaxis
