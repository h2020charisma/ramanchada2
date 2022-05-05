#!/usr/bin/env python3

import pandas

from ramanchada2.io.simulated.vasp.vasp_simulation_dat import lines_from_vasp_dat


def test_lines_from_vasp_dat(vasp_simulation_dat_file):
    df = lines_from_vasp_dat(vasp_simulation_dat_file)
    df = df.apply(pandas.to_numeric)
    df = df.dropna(axis=0).dropna(axis=1)
    assert df.shape == (30, 7)
