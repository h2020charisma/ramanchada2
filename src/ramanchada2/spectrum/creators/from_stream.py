import io
import os
import shutil
import tempfile
from typing import Literal, Optional, Union

import spc_io
from pydantic import validate_call

from ramanchada2.io.experimental import (rc1_parser, read_csv, read_spe,
                                         read_txt)
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ramanchada2.misc.types import SpeMetadataModel

from ..spectrum import Spectrum


@add_spectrum_constructor()
@validate_call(config=dict(arbitrary_types_allowed=True))
def from_stream(in_stream: Union[io.TextIOBase, io.BytesIO, io.BufferedReader],
                filetype: Union[None, Literal['spc', 'sp', 'spa', '0', '1', '2',
                                              'wdf', 'ngs', 'jdx', 'dx',
                                              'txt', 'txtr', 'tsv', 'csv', 'prn', 'dpt',
                                              'rruf', 'spe']],
                filename: Optional[str] = None,
                backend: Union[None, Literal['native', 'rc1_parser']] = None,
                ):
    def load_native():
        if filetype in {'txt', 'txtr', 'tsv', 'prn', 'rruf', 'dpt'}:
            if isinstance(in_stream, io.TextIOBase):
                fp = in_stream
            else:
                fp = io.TextIOWrapper(in_stream)
            x, y, meta = read_txt(fp)
        elif filetype in {'csv'}:
            if isinstance(in_stream, io.TextIOBase):
                fp = in_stream
            else:
                fp = io.TextIOWrapper(in_stream)
            x, y, meta = read_csv(fp)
        elif filetype in {'spc'}:
            if isinstance(in_stream, io.TextIOBase):
                raise ValueError('For spc filetype does not support io.TextIOBase')
            fp = in_stream
            spc = spc_io.SPC.from_bytes_io(fp)
            if len(spc) != 1:
                raise ValueError(f'Single subfile SPCs are supported. {len(spc)} subfiles found')
            x = spc[0].xarray
            y = spc[0].yarray
            meta = spc.log_book.text
        elif filetype in {'spe'}:
            if isinstance(in_stream, io.TextIOBase):
                raise ValueError('For spc filetype does not support io.TextIOBase')
            with tempfile.TemporaryDirectory(suffix='ramanchada2') as dn:
                fn = os.path.basename(filename or in_stream.name or f'noname.{filetype}')
                path = os.path.join(dn, fn)
                with open(path, 'wb') as fp:
                    shutil.copyfileobj(in_stream, fp)
                    print(f'shutil.copyfileobj({in_stream}, {fp}')
                x, y, meta = read_spe(path)
            spe = Spectrum(x=x, y=y, metadata=meta)
        else:
            raise ValueError(f'filetype {filetype} not supported')
        meta["Original file"] = os.path.basename(filename) if filename else 'N/A loaded from stream'
        spe = Spectrum(x=x, y=y, metadata=meta)  # type: ignore
        return spe

    def load_rc1():
        with tempfile.TemporaryDirectory(suffix='ramanchada2') as dn:
            fn = os.path.basename(filename or in_stream.name or f'noname.{filetype}')
            path = os.path.join(dn, fn)
            if isinstance(in_stream, io.TextIOBase):
                with open(path, 'w') as fp:
                    shutil.copyfileobj(in_stream, fp)
                    print(f'shutil.copyfileobj({in_stream}, {fp}')
            else:
                with open(path, 'wb') as fp:
                    shutil.copyfileobj(in_stream, fp)
                    print(f'shutil.copyfileobj({in_stream}, {fp}')
            x, y, meta = rc1_parser.parse(path, filetype)
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
