#!/usr/bin/env python3

from typing import List, Tuple, Dict

import pandas


def bw_format(lines: List[str]) -> Tuple[pandas.DataFrame, Dict]:
    section_split = 0
    for i, ll in enumerate(lines):
        if ll.count(';') > 1:
            section_split = i
            break
    meta = lines[:section_split]
    spec = lines[section_split:]

    meta_dict = dict([m.replace(',', '.').strip().split(';') for m in meta])

    spec_split = [s.replace(',', '.').replace(' ', '').strip('\r\n ;').split(';') for s in spec]
    spe_parsed = pandas.DataFrame.from_records(data=spec_split[1:], columns=spec_split[0])
    spe_parsed = spe_parsed.apply(pandas.to_numeric, errors='coerce')
    spe_parsed = spe_parsed.dropna(axis=0)

    return spe_parsed, meta_dict
