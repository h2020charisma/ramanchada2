#!/usr/bin/env python3
"""Create spectrum from local files."""

from typing import Literal, Union

from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.types import SpeMetadataModel
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ramanchada2.io.experimental import read_txt, read_csv, rc1_parser


@add_spectrum_constructor()
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_local_file(
        in_file_name: str,
        filetype: Union[None, Literal['spc', 'sp', 'spa', '0', '1', '2',
                                      'wdf', 'ngs', 'jdx', 'dx',
                                      'txt', 'txtr', 'csv', 'prn', 'rruf']] = None,
        backend: Literal['native', 'rc1_parser'] = 'rc1_parser'):
    """
    Read experimental spectrum from a local file.

    Parameters
    ----------
    in_file_name : str
        path to a local file, containing a spectrum
    filetype : optional, default is None
        Specify the filetype. Filetype can be any of:
        `spc`, `sp`, `spa`, `0`, `1`, `2`, `wdf`, `ngs`, `jdx`,
        `dx`, `txt`, `txtr`, `csv`, `prn`, `rruf` or `None`
        `None` used to determine by extension of the file.
    backend : Literal['native', 'rc1_parser'], default 'native'

    Raises
    ------
    ValueError
        When called with unsupported file formats
    """
    if backend == 'native':
        if filetype in {'txt', 'txtr', 'prn', 'rruf'}:
            with open(in_file_name) as fp:
                x, y, meta = read_txt(fp)
                spe = Spectrum(x=x, y=y, metadata=meta)  # type: ignore
        elif filetype in {'csv'}:
            with open(in_file_name) as fp:
                x, y, meta = read_csv(fp)
                spe = Spectrum(x=x, y=y, metadata=meta)  # type: ignore
        else:
            raise ValueError(f'filetype {filetype} not supported')
    elif backend == 'rc1_parser':
        x, y, meta = rc1_parser.parse(in_file_name, filetype)
        spe = Spectrum(x=x, y=y, metadata=SpeMetadataModel.parse_obj(meta))
    spe._sort_x()
    return spe
