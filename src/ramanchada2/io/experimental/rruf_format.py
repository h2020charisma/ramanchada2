import re
from typing import Dict, List, Tuple

import pandas
from numpy.typing import NDArray

# A real data row is "<number><sep><number>" (RRUFF's x, y body). Real
# RRUFF exports interleave two kinds of non-"##" lines before the actual
# data starts: a blank separator line, and continuation text wrapped from
# the preceding "##KEY=..." value onto a line that itself has no "##"
# prefix (e.g. a multi-line ##ORIENTATION note). Neither is data, so
# "first line not starting with ##" (the original check) lands on
# whichever of those comes first instead of the real first data row --
# confirmed against real downloaded rruff.net corpus files, where this
# was true for the large majority (blank-line case) and a smaller but
# non-trivial fraction (wrapped-continuation case). "RAW" files (as
# opposed to "Processed") additionally use scientific notation on BOTH
# columns (e.g. "8.758203e+001, 5.919380e+002"), so the position column
# needs the same e/+/- characters as the intensity column, not just
# digits and a decimal point. A minority of RAW files use a tab, or (rarer
# still) plain whitespace, instead of a comma as the column separator --
# also confirmed against the real corpus -- so all three are accepted.
_NUMBER = r'-?[0-9.]+(?:[eE][+-]?[0-9]+)?'
_SEP = r'(?:,|\t|[ \t]+)'
_DATA_ROW = re.compile(r'^\s*{n}\s*{sep}\s*{n}\s*$'.format(n=_NUMBER, sep=_SEP))
# One split per row: a comma (with any surrounding whitespace collapsed
# into it) or a run of whitespace -- NOT "comma or whitespace" matched
# independently, which would treat "1.0, 2.0"'s comma AND its following
# space as two separate splits and produce 3 fields instead of 2.
_SPLIT_SEP = re.compile(r'\s*,\s*|\s+')


def rruf_format(lines: List[str]) -> Tuple[NDArray, NDArray, Dict]:
    header_lines = []
    start_spe = None
    for i, ll in enumerate(lines):
        if ll.startswith('##'):
            header_lines.append(ll)
        elif _DATA_ROW.match(ll):
            start_spe = i
            break
        # else: blank separator or wrapped header continuation -- skip,
        # keep looking for the first genuine data row.
    if start_spe is None:
        raise ValueError('rruf_format: no numeric data rows found')
    meta = dict([ll.strip()[2:].split('=', 1) for ll in header_lines])
    stop_spe = len(lines)
    for i, ll in enumerate(lines):
        if ll.startswith('##END'):
            stop_spe = i
            break
    data = pandas.DataFrame.from_records(
        data=[_SPLIT_SEP.split(ll.strip()) for ll in lines[start_spe:stop_spe] if ll.strip()]
        ).apply(pandas.to_numeric).dropna(axis=0)
    positions, intensities = data.to_numpy().T
    return positions, intensities, meta
