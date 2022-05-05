import pandas

from ramanchada2.io.simulated.crystal.discrete_lines_dat import lines_from_crystal_dat


def test_lines_from_crystal_dat_raw(crystal_simulation_raw_dat_file):
    df = lines_from_crystal_dat(crystal_simulation_raw_dat_file)
    df = df.apply(pandas.to_numeric)
    df = df.dropna(axis=0).dropna(axis=1)
    assert df.shape == (5, 10), 'wrong shape'


def test_lines_from_crystal_dat_convolved(crystal_simulation_convolved_dat_file):
    df = lines_from_crystal_dat(crystal_simulation_convolved_dat_file)
    df = df.apply(pandas.to_numeric)
    df = df.dropna(axis=0).dropna(axis=1)
    assert df.shape == (1809, 10), 'wrong shape'
