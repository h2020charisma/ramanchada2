from typing import Dict, List, Tuple

import pandas
from numpy.typing import NDArray


def rruf_format(lines: List[str]) -> Tuple[NDArray, NDArray, Dict]:
    for i, ll in enumerate(lines):
        if not ll.startswith('##'):
            start_spe = i
            break
    meta = dict([ll.strip()[2:].split('=') for ll in lines[:start_spe]])
    for i, ll in enumerate(lines):
        if ll.startswith('##END'):
            stop_spe = i
            break
    data = pandas.DataFrame.from_records(
        data=[ll.split(',') for ll in lines[start_spe:stop_spe]]
        ).apply(pandas.to_numeric).dropna(axis=0)
    positions, intensities = data.to_numpy().T
    return positions, intensities, meta
