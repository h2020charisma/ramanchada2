import numpy as np

import ramanchada2 as rc2


def test_empty_spectrum():
    spe = rc2.spectrum.Spectrum(x=np.arange(100), y=np.zeros(100))
    spe.find_peak_multipeak()
