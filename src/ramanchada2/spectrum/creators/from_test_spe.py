#!/usr/bin/env python3
"""Create spectrum from local files."""

import random

from ramanchada2.auxiliary.spectra.datasets2 import (get_filenames,
                                                     prepend_prefix)
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor

from ..spectrum import Spectrum


@add_spectrum_constructor()
def from_test_spe(index=None, **kwargs):
    """Create new spectrum from test data.

    Args:
        index:
            `int` or `None`, optional, default is `None`. If `int`: will be used as an index of filtered list. If
            `None`: a random spectrum will be taken.
        **kwargs:
            The rest of the parameters will be used as filter.
    """
    filtered = prepend_prefix(get_filenames(**kwargs))
    if index is None:
        fn = random.sample(filtered, 1)[0]
    else:
        fn = filtered[index]
    spe = Spectrum.from_local_file(fn)
    return spe
