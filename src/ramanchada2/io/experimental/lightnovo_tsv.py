from typing import Dict, List, Tuple

import pandas


def lightnovo_tsv(lines: List[str]) -> Tuple[pandas.DataFrame, Dict]:
    for i, ll in enumerate(lines):
        if not ll.startswith(('DeviceSN', 'LaserWavelength_nm', 'Exposure_ms',
                              'GainMultiplier', 'LaserCurrent_mA', 'LaserPower_mW',
                              'RepetitionCount', 'Tags')):
            start_spe = i
            break
    else:
        raise ValueError('The input is not lightnovo tsv format')
    meta = dict([i.split('\t', 1) for i in lines[:start_spe]])
    spe_lines = [ll.strip().split('\t') for ll in lines[start_spe:]]

    data = pandas.DataFrame.from_records(data=spe_lines, columns=['Position', 'Amplitude']
                                         ).apply(pandas.to_numeric, errors='coerce'
                                                 ).dropna(axis=0)
    return data, meta
