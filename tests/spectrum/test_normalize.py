from math import isclose

import numpy as np

import ramanchada2 as rc2


def test_normalize():
    spe = rc2.spectrum.from_test_spe()

    tt = spe.normalize(strategy='unity').y
    assert isclose(np.sum(tt), 1)

    tt = spe.normalize(strategy='min_unity').y
    assert isclose(np.sum(tt), 1)
    assert np.min(tt) == 0

    tt = spe.normalize(strategy='unity_density').y
    assert np.allclose(spe.normalize(strategy='unity_area').y, tt)
    assert isclose(np.sum(np.diff(spe.x)*(tt[1:]+tt[:-1])/2)
                   + tt[0]*(spe.x[1]-spe.x[0])/2
                   + tt[-1]*(spe.x[-1]-spe.x[-2])/2,
                   1)

    tt = spe.normalize(strategy='minmax').y
    assert np.min(tt) == 0
    assert np.max(tt) == 1

    tt = spe.normalize(strategy='L1').y
    assert isclose(np.sum(np.abs(tt)), 1)

    tt = spe.normalize(strategy='L2').y
    assert isclose(np.sum(tt**2), 1)
