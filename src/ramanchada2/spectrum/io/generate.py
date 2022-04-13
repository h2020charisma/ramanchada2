#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from ..spectrum import Spectrum
from ramanchada2.spectral_components.spectral_component_collection import SpectralComponentCollection
from ramanchada2.misc.spectrum_constructor import spectrum_constructor_deco


@spectrum_constructor_deco
def generate(spe: Spectrum,
             spe_components: SpectralComponentCollection,
             x=500, metadata={'origin': 'generated'}):
    """
    spe_components : SpectralComponentCollection
    x : int or array-like, optional, default 500
        x axis of the spectrum
    metadata : dict, optional
    """
    if hasattr(x, '__len__'):
        spe.x = x
    else:
        spe.x = np.arange(x)
    spe.y = spe_components(spe.x)
    spe.meta = metadata
