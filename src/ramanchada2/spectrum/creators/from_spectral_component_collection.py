from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ramanchada2.spectral_components.spectral_component_collection import \
    SpectralComponentCollection

from ..spectrum import Spectrum


@add_spectrum_constructor()
@validate_call(config=dict(arbitrary_types_allowed=True))
def from_spectral_component_collection(
        spe_components: SpectralComponentCollection,
        x=2000):
    """
    from_spectral_component_collection

    Args:
        spe_components:
            SpectralComponentCollection
        x:
            `int` or array-like, optional, default `2000`. `x` axis of the spectrum.
    """

    spe = Spectrum(x=x, metadata={'origin': 'generated'})  # type: ignore
    spe.y = spe_components(spe.x)
    return spe
