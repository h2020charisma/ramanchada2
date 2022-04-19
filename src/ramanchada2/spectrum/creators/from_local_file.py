#!/usr/bin/env python3
"""Create spectrum from local files."""

from __future__ import annotations

from typing import Literal

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_constructor import spectrum_constructor_deco
from ramanchada2.io.experimental import read_txt


@spectrum_constructor_deco
def from_local_file(
        spe: Spectrum,
        in_file_name: str,
        filetype: Literal['txt'] = 'txt',
        **kwargs):
    """
    Read experimental spectrum from a local file.
    """
    if filetype in {'jdx', 'dx'}:
        raise NotImplementedError('The implementation of JCAMP reader is missing')
    elif filetype in {'txt', 'txtr', 'csv', 'prn', 'rruf'}:
        with open(in_file_name) as fp:
            x, y, meta = read_txt(fp)
            spe.x = x
            spe.y = y
            spe.meta = meta
    else:
        raise ValueError(f'filetype {filetype} not supported')
    # Vlues for spe.x, spe.y, spe.meta should be set here
