#!/usr/bin/env python3

from typing import TextIO, Tuple, Dict

from numpy.typing import NDArray

from .bw_format import bw_format
from .two_column_spe import two_column_spe


""" There are 4 types of TXT data files that can be distinguished by their first line:
    1. File Version;BWRam4.11_11
    2. File Version;BWSpec4.11_1
    3. <wavenumber>	<intensity>
    4. #Wave		#Intensity
      <wavenumber>	<intensity>
    """


def read_txt(data_in: TextIO) -> Tuple[NDArray, NDArray, Dict]:
    lines = data_in.readlines()
    if lines[0].startswith('File Version;BW'):
        data, meta = bw_format(lines)
        positions = data['RamanShift'].to_numpy()
        intensities = data['DarkSubtracted#1'].to_numpy()
    else:  # assume two column spectrum
        positions, intensities = two_column_spe(lines)
        meta = dict()
    return positions, intensities, meta
