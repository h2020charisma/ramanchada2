import numpy as np

import ramanchada2 as rc2


def test_dropna():
    x = np.array([0, 1, 2, np.nan, np.nan, 3, 4, 5, 6, np.nan])
    y = np.array([np.nan, 1, 2, 3, np.nan, np.nan, 4, 5, 6, np.nan])
    spe = rc2.spectrum.Spectrum(x=np.array([1]), y=np.array([2]))
    spe.x = x
    spe.y = y

    spe_new = spe.dropna()

    assert np.all(spe_new.x == spe_new.y)
    assert np.all(spe_new.x == np.array([1, 2, 4, 5, 6]))
