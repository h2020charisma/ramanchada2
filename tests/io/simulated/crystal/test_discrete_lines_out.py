from ramanchada2.io.simulated.crystal.discrete_lines_out import lines_from_crystal_out


def test_lines_from_crystal_out(crystal_simulation_out_file):
    df = lines_from_crystal_out(crystal_simulation_out_file)
    df = df.dropna(axis=0).dropna(axis=1)
    assert df.shape == (6, 13)
