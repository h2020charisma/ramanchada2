import numpy as np
import pytest

import ramanchada2 as rc2


def test_spectrum_init():
    with pytest.raises(ValueError):
        spe = rc2.spectrum.Spectrum(x=np.arange(100), y=np.arange(101))
    with pytest.raises(ValueError):
        spe = rc2.spectrum.Spectrum(x=np.arange(100), y=np.arange(99))
    spe = rc2.spectrum.Spectrum(x=np.arange(100, dtype=int), y=np.random.uniform(size=100))
    assert spe.x.dtype == float
    assert spe.y.dtype == float
    spec = spe.__copy__()
    assert np.all(spe.x == spec.x)
    assert np.all(spe.y == spec.y)
    assert spe._xdata is spec._xdata
    assert spe._ydata is spec._ydata

    spe = rc2.spectrum.Spectrum(x=np.arange(100, dtype=int))
    assert spe._ydata is None
    with pytest.raises(ValueError):
        spe.y
