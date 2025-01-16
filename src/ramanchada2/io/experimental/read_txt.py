from typing import Dict, TextIO, Tuple

import numpy as np
from numpy.typing import NDArray

from .bw_format import bw_format
from .neegala_format import neegala_format
from .lightnovo_tsv import lightnovo_tsv
from .rruf_format import rruf_format

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
        meta['@axes'] = ['RamanShift']
        meta['@signal'] = 'DarkSubtracted'
    elif lines[0].startswith('##'):
        # rruf format
        positions, intensities, meta = rruf_format(lines)
        meta['@axes'] = ['']
        meta['@signal'] = ''
    elif ',' in lines[0] and not lines[0].split(',')[0].isdigit():
        data, meta = neegala_format(lines)
        positions = data['Raman_Shift'].to_numpy()
        intensities = data['Processed_Data'].to_numpy()
        meta['@axes'] = ['Raman_Shift']
        meta['@signal'] = 'Processed_Data'
    elif lines[0].startswith('DeviceSN\t'):
        data, meta = lightnovo_tsv(lines)
        positions = data['Position'].to_numpy()
        intensities = data['Amplitude'].to_numpy()
        meta['@axes'] = ['']
        meta['@signal'] = ''
    else:  # assume two column spectrum
        meta = dict()
        if lines[0].startswith('#'):
            # assume header row
            data = np.genfromtxt(lines, names=True, loose=False)
            meta['@axes'] = [data.dtype.names[0]]
            meta['@signal'] = data.dtype.names[1]
            data = np.array(data.tolist())
        else:
            data = np.genfromtxt(lines, loose=False)
            meta['@axes'] = ['']
            meta['@signal'] = ''
        positions, intensities = data.T
    return positions, intensities, meta
