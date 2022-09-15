#!/usr/bin/env python3
"""Create spectrum from local files."""

from typing import Literal

from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.types import SpeMetadataModel
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ramanchada2.io.experimental import read_txt, read_csv


@add_spectrum_constructor()
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_local_file(
        in_file_name: str,
        filetype: Literal['txt', 'csv'] = 'txt',
        backend: Literal['native', 'ramanchada_parser'] = 'native'):
    """
    Read experimental spectrum from a local file.

    Parameters
    ----------
    in_file_name : str
        path to a local file, containing a spectrum
    filetype : Literal['txt', 'csv'], optional
        filetype of the file. For the timebeing only `.txt` files are supported, by default 'txt'
    backend : Literal['native', 'ramanchada-parser'], default 'native'

    Raises
    ------
    NotImplementedError
        When called with unsupported file formats
    ValueError
        When called with unsupported file formats
    """
    if backend == 'native':
        if filetype in {'jdx', 'dx'}:
            raise NotImplementedError('The implementation of JCAMP reader is missing')
        elif filetype in {'txt', 'txtr', 'prn', 'rruf'}:
            with open(in_file_name) as fp:
                x, y, meta = read_txt(fp)
                spe = Spectrum(x=x, y=y, metadata=meta)  # type: ignore
        elif filetype in {'csv'}:
            with open(in_file_name) as fp:
                x, y, meta = read_csv(fp)
                spe = Spectrum(x=x, y=y, metadata=meta)  # type: ignore
        else:
            raise ValueError(f'filetype {filetype} not supported')
    elif backend == 'ramanchada_parser':
        from ramanchada_parser import parse_file
        x, y, meta = parse_file(in_file_name)
        spe = Spectrum(x=x, y=y, metadata=SpeMetadataModel.parse_obj(meta))
    spe._sort_x()
    return spe
