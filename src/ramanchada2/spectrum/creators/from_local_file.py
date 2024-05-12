#!/usr/bin/env python3
"""Create spectrum from local files."""

from typing import Literal, Union

from pydantic import validate_arguments

import os
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
        backend: Union[None, Literal['native', 'rc1_parser']] = None):
    """
    Read experimental spectrum from a local file.

    Args:
        in_file_name:
            Path to a local file containing a spectrum.
        filetype:
            Specify the filetype. Filetype can be any of: `spc`, `sp`, `spa`, `0`, `1`, `2`, `wdf`, `ngs`, `jdx`, `dx`,
            `txt`, `txtr`, `csv`, `prn`, `rruf` or `None`. `None` used to determine by extension of the file.
        backend:
            `native`, `rc1_parser` or `None`. `None` means both.

    Raises:
        ValueError:
            When called with unsupported file formats.
    """
    def load_native():
        if filetype is None:
            ft = os.path.splitext(in_file_name)[1][1:]
        else:
            ft = filetype
        if ft in {'txt', 'txtr', 'prn', 'rruf'}:
            with open(in_file_name) as fp:
                x, y, meta = read_txt(fp)
        elif ft in {'csv'}:
            with open(in_file_name) as fp:
                x, y, meta = read_csv(fp)
        else:
            raise ValueError(f'filetype {ft} not supported')
        meta["Original file"] = os.path.basename(in_file_name)
        spe = Spectrum(x=x, y=y, metadata=meta)  # type: ignore
        return spe

    def load_rc1():
        x, y, meta = rc1_parser.parse(in_file_name, filetype)
        spe = Spectrum(x=x, y=y, metadata=SpeMetadataModel.parse_obj(meta))
        return spe

    if backend == 'native':
        spe = load_native()
    elif backend == 'rc1_parser':
        spe = load_rc1()
    elif backend is None:
        try:
            spe = load_native()
        except Exception:
            spe = load_rc1()
    spe._sort_x()
    return spe
