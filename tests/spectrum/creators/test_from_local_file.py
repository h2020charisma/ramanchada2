import numpy as np

import ramanchada2 as rc2


def test_from_local_file(experimental_filename):
    if experimental_filename[-4:] in {'.l6s', '.wxd'}:
        return
    elif experimental_filename[-4:] in {'.wdf'}:
        spe = rc2.spectrum.from_local_file(experimental_filename)
    else:
        spe = rc2.spectrum.from_local_file(experimental_filename, backend='native')
    assert len(spe.meta.__root__) > 0
    assert np.all(np.isfinite(spe.x))
    assert np.all(np.isfinite(spe.y))
    assert len(spe.x) == len(spe.y)
    assert len(spe.x) > 0
    assert np.all(np.diff(spe.x) > 0)
