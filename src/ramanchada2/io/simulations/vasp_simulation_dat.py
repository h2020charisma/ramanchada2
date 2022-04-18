#!/usr/bin/env python3

from io import TextIOBase

import pandas
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def lines_from_vasp_dat(data_in: TextIOBase) -> pandas.DataFrame:
    lines = data_in.readlines()
    lines_split = [ll.strip(' \r\n#').split() for ll in lines]
    df = pandas.DataFrame.from_records(data=lines_split[1:], columns=lines_split[0])
    df = df.apply(pandas.to_numeric)
    return df
