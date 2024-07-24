import numpy as np

import ramanchada2 as rc2


def test_from_local_file_rruf(rruf_experimental_filename):
    spe = rc2.spectrum.from_local_file(rruf_experimental_filename)
    assert len(spe.meta.__root__) > 0
    assert np.all(np.isfinite(spe.x))
    assert np.all(np.isfinite(spe.y))
    assert len(spe.x) == len(spe.y)
    assert len(spe.x) > 0
