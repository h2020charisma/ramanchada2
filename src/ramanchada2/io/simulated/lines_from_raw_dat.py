from io import TextIOBase

import pandas as pd
from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def lines_from_raw_dat(data_in: TextIOBase) -> pd.DataFrame:
    df = pd.read_table(data_in, sep=r'\s+', dtype=float)
    df.rename(columns={'#FREQUENCIES': 'Frequencies'}, inplace=True)
    return df
