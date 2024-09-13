from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter

from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def scale_yaxis_linear(old_spe: Spectrum,
                       new_spe: Spectrum,
                       factor: float = 1):
    """
    Scale y-axis values

    This function provides the same result as `spe*const`

    Args:
        old_spe: internal use only
        new_spe: internal use only
        factor optional. Defaults to 1.
            Y-values scaling factor

    Returns: corrected spectrum
    """
    new_spe.y = old_spe.y * factor
