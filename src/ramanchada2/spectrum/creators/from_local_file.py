"""Create spectrum from local files."""

import os
from typing import Literal, Union, Dict

import spc_io
from pydantic import validate_call

from ramanchada2.io.experimental import (rc1_parser, read_csv, read_spe,
                                         read_txt)
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ramanchada2.misc.types import SpeMetadataModel

from ..spectrum import Spectrum
from .from_chada import from_chada


@add_spectrum_constructor()
@validate_call(config=dict(arbitrary_types_allowed=True))
def from_local_file(
        in_file_name: str,
        filetype: Union[None, Literal['spc', 'sp', 'spa', '0', '1', '2',
                                      'wdf', 'ngs', 'jdx', 'dx',
                                      'txt', 'txtr', 'tsv', 'csv', 'prn', 'dpt',
                                      'rruf', 'spe', 'cha']] = None,
        backend: Union[None, Literal['native', 'rc1_parser']] = None,
        custom_meta: Dict = {}):
    """
    Read experimental spectrum from a local file.

    Args:
        in_file_name:
            Path to a local file containing a spectrum.
        filetype:
            Specify the filetype. Filetype can be any of: `spc`, `sp`, `spa`, `0`, `1`, `2`, `wdf`, `ngs`, `jdx`, `dx`,
            `txt`, `txtr`, `tsv`, `csv`, `dpt`, `prn`, `rruf`, `spe` (Princeton Instruments) or `None`.
            `None` used to determine by extension of the file.
        backend:
            `native`, `rc1_parser` or `None`. `None` means both.
        custom_meta: Dict
            Add custom metadata fields in the created spectrum

    Raises:
        ValueError:
            When called with unsupported file formats.
    """
    def load_native():
        if filetype is None:
            ft = os.path.splitext(in_file_name)[1][1:]
        else:
            ft = filetype
        if ft in {'cha'}:
            return from_chada(filename=in_file_name)
        elif ft in {'txt', 'txtr', 'prn', 'rruf', 'tsv', 'dpt'}:
            with open(in_file_name) as fp:
                x, y, meta = read_txt(fp)
        elif ft in {'csv'}:
            with open(in_file_name) as fp:
                x, y, meta = read_csv(fp)
        elif ft in {'spc'}:
            with open(in_file_name, 'rb') as fp:
                spc = spc_io.SPC.from_bytes_io(fp)
                if len(spc) != 1:
                    raise ValueError(f'Single subfile SPCs are supported. {len(spc)} subfiles found')
                x = spc[0].xarray
                y = spc[0].yarray
                meta = spc.log_book.text
        elif ft in {'spe'}:
            x, y, meta = read_spe(in_file_name)
        else:
            raise ValueError(f'filetype {ft} not supported')
        meta["Original file"] = os.path.basename(in_file_name)
        meta.update(custom_meta)
        spe = Spectrum(x=x, y=y, metadata=meta)  # type: ignore
        return spe

    def load_rc1():
        x, y, meta = rc1_parser.parse(in_file_name, filetype)
        spe = Spectrum(x=x, y=y, metadata=SpeMetadataModel.model_validate(meta))
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
