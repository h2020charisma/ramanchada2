from typing import Dict, List, Tuple

import pandas


def neegala_format(lines: List[str]) -> Tuple[pandas.DataFrame, Dict]:
    for i, ll in enumerate(lines):
        if ll.startswith('Pixels,Wavelength,Wavenumbers,Raman_Shift,Raw_Data,Background_Data,Processed_Data'):
            start_spe = i
            break
    else:
        raise ValueError('The input is not neegala format')
    meta = dict([i.split(',', 1) for i in lines[:start_spe]])
    spe_lines = [ll.strip().split(',') for ll in lines[start_spe:]]

    data = pandas.DataFrame.from_records(data=spe_lines[1:], columns=spe_lines[0]
                                         ).apply(pandas.to_numeric, errors='coerce'
                                                 ).dropna(axis=0)
    return data, meta
