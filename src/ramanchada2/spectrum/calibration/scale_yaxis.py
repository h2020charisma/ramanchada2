from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ..spectrum import Spectrum


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def scale_yaxis_linear(old_spe: Spectrum,
                       new_spe: Spectrum,
                       factor: float = 1):
    new_spe.y = old_spe.y * factor
