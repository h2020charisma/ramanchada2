from io import TextIOBase

import pandas
from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def lines_from_vasp_dat(data_in: TextIOBase) -> pandas.DataFrame:
    """
    calculates perpendicular and parallel intensities using
    https://doi.org/10.1103/PhysRevB.54.7830
    """
    lines = data_in.readlines()
    lines_split = [ll.strip(' \r\n#').split() for ll in lines]
    df = pandas.DataFrame.from_records(data=lines_split[1:], columns=lines_split[0])
    df = df.apply(pandas.to_numeric)

    alpha = df['alpha']
    beta2 = df['beta2']
    perp_par_ratio = 3*beta2/(45*alpha**2 + 4*beta2)
    perp_par_ratio = perp_par_ratio.fillna(0)

    i_tot = df['activity']
    i_perp = i_tot * perp_par_ratio
    i_par = i_tot * (1 - perp_par_ratio)
    df = df.merge(i_par.to_frame(name='I_par'), left_index=True, right_index=True)
    df = df.merge(i_perp.to_frame(name='I_perp'), left_index=True, right_index=True)
    return df
