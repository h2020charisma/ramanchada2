import numpy as np
from scipy.interpolate import CubicSpline

import ramanchada2 as rc2


def test_resample_spline():
    spe = rc2.spectrum.from_test_spe(2, laser_wl=['785'], sample=['PST']).normalize()
    spe_tr = spe.trim_axes(method='x-axis', boundaries=(400, 2000))
    res_spe = spe.resample_spline_filter((400, 2000), 1000, spline='akima', cumulative=False)
    aaa = CubicSpline(spe_tr.x, spe_tr.y)(np.linspace(400, 2000, 1000, endpoint=False))
    assert np.all(res_spe.x == np.linspace(400, 2000, 1000, endpoint=False))
    assert np.allclose(aaa, res_spe.y, rtol=.05)

    spe = rc2.spectrum.from_delta_lines({500: 100, 600: 200})
    spe = spe.add_gaussian_noise_drift(sigma=2, coef=.3)

    for s in 'pchip', 'akima', 'makima', 'cubic_spline':
        tt = spe.resample_spline_filter(x_range=(0, 1000), xnew_bins=1000, spline=s)
        assert np.all(tt.y[:450] == 0)
        assert np.all(tt.y[650:] == 0)
