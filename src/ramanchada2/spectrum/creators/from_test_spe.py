#!/usr/bin/env python3
"""Create spectrum from local files."""

import random

from ramanchada2.auxiliary.spectra.datasets2 import get_filenames
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor

from ..spectrum import Spectrum


@add_spectrum_constructor()
def from_test_spe(index=None, **kwargs):
    """Create new spectrum from test data.

    Parameters
    ----------
    index : int, None, Optional, default is None
        if `index` is int it will be used as an index of filtered list
        if `index` is None a random spectrum will be taken
    **kwargs :
        the rest of the parameters will be used as filter
    """
    filtered = get_filenames(**kwargs)
    if index is None:
        fn = random.sample(filtered, 1)[0]
    else:
        fn = filtered[index]
    spe = Spectrum.from_local_file(fn)
    return spe
