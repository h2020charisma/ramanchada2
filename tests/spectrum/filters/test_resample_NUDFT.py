import numpy as np
from scipy.interpolate import CubicSpline

import ramanchada2 as rc2


def test_no_resampling():
    spe0 = rc2.spectrum.from_delta_lines(deltas={100: 100, 200: 200, 300: 300}, xcal=lambda x: x, nbins=400)
    spe1 = spe0.resample_NUDFT_filter(x_range=(0, 400), xnew_bins=400, window=lambda x: [1]*x)
    spe2 = spe0.resample_NUDFT_filter(x_range=(50, 400), xnew_bins=350, window=lambda x: [1]*x)
    assert np.allclose(spe0.y, spe1.y)
    assert np.allclose(spe0.y[(spe0.x >= 50) & (spe0.x < 400)], spe2.y)


def test_resampling():
    fn = rc2.auxiliary.spectra.datasets2.prepend_prefix(['FMNT-M_BW532/PST10_iR532_Probe_100_3000msx7.txt'])[0]
    spe0 = rc2.spectrum.from_local_file(fn).subtract_moving_minimum(10).normalize()
    spe1 = spe0.resample_NUDFT_filter(x_range=(300, 3500), xnew_bins=1000, window='blackmanharris')

    x0 = spe0.x
    y0 = spe0.y
    y0 = y0[(x0 >= 300) & (x0 < 3500)]
    x0 = x0[(x0 >= 300) & (x0 < 3500)]

    x1 = spe1.x
    y1 = spe1.y

    cs0 = CubicSpline(x0, np.cumsum(y0))
    cs1 = CubicSpline(x1, np.cumsum(y1))

    assert np.allclose(cs0(x0[40:]), cs1(x0[40:]), rtol=7e-2)
