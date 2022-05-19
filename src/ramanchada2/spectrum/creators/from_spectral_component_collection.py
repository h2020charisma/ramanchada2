#!/usr/bin/env python3

from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.spectral_components.spectral_component_collection import SpectralComponentCollection
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor


@add_spectrum_constructor()
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_spectral_component_collection(
        spe_components: SpectralComponentCollection,
        x=2000):
    """
    spe_components : SpectralComponentCollection
    x : int or array-like, optional, default 2000
        x axis of the spectrum
    metadata : dict, optional
    """

    spe = Spectrum(x=x, metadata={'origin': 'generated'})  # type: ignore
    spe.y = spe_components(spe.x)
    return spe
