#!/usr/bin/env python3
"""Create spectrum from local files."""

from typing import Literal

from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ramanchada2.io.experimental import read_txt


@add_spectrum_constructor()
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_local_file(
        in_file_name: str,
        filetype: Literal['txt'] = 'txt'):
    """
    Read experimental spectrum from a local file.

    Parameters
    ----------
    in_file_name : str
        path to a local file, containing a spectrum
    filetype : Literal[&#39;txt&#39;], optional
        filetype of the file. For the timebeing only `.txt` files are supported, by default 'txt'

    Raises
    ------
    NotImplementedError
        When called with unsupported file formats
    ValueError
        When called with unsupported file formats
    """
    if filetype in {'jdx', 'dx'}:
        raise NotImplementedError('The implementation of JCAMP reader is missing')
    elif filetype in {'txt', 'txtr', 'csv', 'prn', 'rruf'}:
        with open(in_file_name) as fp:
            x, y, meta = read_txt(fp)
            return Spectrum(x=x, y=y, metadata=meta)  # type: ignore
    else:
        raise ValueError(f'filetype {filetype} not supported')
