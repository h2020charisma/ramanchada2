#!/usr/bin/env python3

import pydantic

from ramanchada2.io.HSDS import read_cha
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ..spectrum import Spectrum


@add_spectrum_constructor(set_applied_processing=False)
@pydantic.validate_arguments
def from_chada(filename: str, dataset: str = '/raw', h5module=None):
    x, y, meta = read_cha(filename, dataset, h5module=h5module)
    return Spectrum(x=x, y=y, metadata=meta)  # type: ignore
