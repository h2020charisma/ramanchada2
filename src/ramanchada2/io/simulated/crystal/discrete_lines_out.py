import re
from io import TextIOBase
from typing import List

import pandas as pd
from pydantic import validate_call

from ramanchada2.misc.exceptions import InputParserError


@validate_call(config=dict(arbitrary_types_allowed=True))
def lines_from_crystal_out(data_in: TextIOBase) -> pd.DataFrame:
    def advance_to(content: str) -> None:
        while content not in data_in.readline():
            continue

    def read_paragraph() -> List[str]:
        ret = list()
        while True:
            line = data_in.readline().rstrip()
            if line == '':
                break
            ret.append(line)
        return ret

    def parse(regex, lines: List[str]) -> List[List[str]]:
        ret = list()
        if len(lines) < 2:
            raise InputParserError()
        for match in [regex.match(line) for line in lines[2:]]:
            if match:
                ret.append(match.groups())
            else:
                raise InputParserError()
        return ret

    def skip_line():
        data_in.readline()

    advance_to('POLYCRYSTALLINE ISOTROPIC INTENSITIES (ARBITRARY UNITS)')
    skip_line()  # empty line
    polyXtal_lines = read_paragraph()
    advance_to('SINGLE CRYSTAL DIRECTIONAL INTENSITIES (ARBITRARY UNITS)')
    skip_line()  # empty line
    monoXtal_lines = read_paragraph()
    # data_in is processed

    polyXtal_regex = re.compile(r'\s*(\d+)-\s*(\d+)\s*([\d.]+)\s*\((\w+)\s*\)' + r'\s*([\d.]+)'*3)
    polyXtal_parsed = parse(polyXtal_regex, polyXtal_lines)
    polyXtal_df = pd.DataFrame.from_records(
        polyXtal_parsed,
        columns=['ModeL', 'ModeU', 'Frequencies', 'Origin', 'I_tot', 'I_par', 'I_perp'])
    polyXtal_df = polyXtal_df.astype(
        dict(zip(polyXtal_df.keys(), [*[int]*2, float, str, *[float]*3])))

    monoXtal_regex = re.compile(r'\s*(\d+)-\s*(\d+)\s*([\d.]+)\s*\((\w+)\s*\)' + r'\s*([\d.]+)'*6)
    monoXtal_parsed = parse(monoXtal_regex, monoXtal_lines)
    monoXtal_df = pd.DataFrame.from_records(
        monoXtal_parsed,
        columns=['ModeL', 'ModeU', 'Frequencies', 'Origin', 'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz'])
    monoXtal_df = monoXtal_df.astype(
        dict(zip(monoXtal_df.keys(), [*[int]*2, float, str, *[float]*6])))
    merge = pd.merge(polyXtal_df, monoXtal_df)
    return merge
