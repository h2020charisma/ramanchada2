from collections import namedtuple
from typing import Dict

import spc_io

from ...misc.types import SpeMetadataModel
from ..spectrum import Spectrum

SPCMapCoordinates = namedtuple('SPCMapCoordinates', ['z', 'w'])


def read_map_spc(filename: str) -> Dict[SPCMapCoordinates, Spectrum]:
    spc = spc_io.SPC.from_bytes_io(open(filename, 'rb'))

    ret = dict()

    spc_meta = {k.strip(): v.strip() for k, v in spc.log_book.text.items()}
    for meas in spc:
        spe_meta = {}
        spe_meta.update(spc_meta)
        spe_meta.update(dict(w=meas.w, z=meas.z))
        ret[SPCMapCoordinates(w=meas.w, z=meas.z)] = Spectrum(x=meas.xarray, y=meas.yarray,
                                                              metadata=SpeMetadataModel.model_validate(spe_meta))
    return ret
