#!/usr/bin/env python3

from typing import List, Tuple

import pandas
import pydantic

from ramanchada2.misc.types import SpectrumMetaData


def bw_format(lines: List[str]) -> Tuple[pandas.DataFrame, SpectrumMetaData]:
    section_split = 0
    for i, ll in enumerate(lines):
        if ll.count(';') > 1:
            section_split = i
            break
    meta = lines[:section_split]
    spec = lines[section_split:]

    meta_split = dict([m.replace(',', '.').strip().split(';') for m in meta])
    meta_parsed = pydantic.parse_obj_as(SpectrumMetaData, meta_split)

    spec_split = [s.replace(',', '.').replace(' ', '').strip('\r\n ;').split(';') for s in spec]
    spe_parsed = pandas.DataFrame.from_records(data=spec_split[1:], columns=spec_split[0])
    spe_parsed = spe_parsed.apply(pandas.to_numeric)
    spe_parsed = spe_parsed.dropna(axis=0)

    return spe_parsed, meta_parsed
