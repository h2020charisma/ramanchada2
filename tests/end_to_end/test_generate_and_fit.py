import numpy as np
import pandas as pd

import ramanchada2 as rc2
from ramanchada2 import spectrum


def test_generate_and_fit():
    lines = {40: 20, 150: 15, 200: 30, 500: 50, 550: 5}
    shift = 50
    spe = spectrum.from_delta_lines(lines).convolve('voigt', sigma=3)
    spe = spe.add_baseline(n_freq=5, amplitude=3, pedestal=0, rng_seed=1111)
    spe = spe.add_gaussian_noise(.1, rng_seed=1111)
    spe = spe.scale_xaxis_fun(lambda x: x - shift)
    candidates = spe.find_peak_multipeak(prominence=spe.y_noise*5, wlen=40, sharpening=None)

    true_pos = np.array(list(lines.keys()))
    calc_pos = [i for gr in candidates for i in gr.positions]
    fit_peaks = spe.fit_peak_multimodel(profile='Moffat', candidates=candidates)
    fit_pos = fit_peaks.locations

    assert len(true_pos) == len(calc_pos), 'wrong number of peaks found'
    assert np.max(np.abs(true_pos - fit_pos - shift)) < 3, 'fit locations far from truth'
    assert np.max(np.abs(true_pos - calc_pos - shift)) < 5, 'find_peaks locations far from truth'


def test_gaussian_fit_parameters():
    x = np.linspace(-100, 3500, 2500)
    params = [
        dict(center=400, sigma=6, amplitude=500),
        dict(center=430, sigma=3, amplitude=300),
        dict(center=600, sigma=10, amplitude=500),
        dict(center=625, sigma=5, amplitude=300),
        dict(center=700, sigma=10, amplitude=500),
        dict(center=780, sigma=5, amplitude=300),
    ]
    spe = rc2.spectrum.from_theoretical_lines(shapes=['gaussian']*len(params),
                                              params=params, x=x)
    spe = spe.add_gaussian_noise_drift(sigma=.5, coef=.2)
    spe = spe.subtract_moving_minimum(100)
    params_df = pd.DataFrame.from_records(params)

    cand = spe.find_peak_multipeak(prominence=spe.y_noise_MAD()*10, width=3, wlen=100)
    fitres = spe.fit_peak_multimodel(profile='Gaussian', candidates=cand, vary_baseline=True)
    df = fitres.to_dataframe_peaks()

    assert np.all(np.isclose(df['amplitude'], params_df['amplitude'], atol=df['amplitude_stderr']*5))
    assert np.all(np.isclose(df['center'], params_df['center'], atol=df['center_stderr']*5))
    assert np.all(np.isclose(df['sigma'], params_df['sigma'], atol=df['sigma_stderr']*5))
    assert np.all(np.isclose(df['sigma']*np.sqrt(8*np.log(2)), df['fwhm']))
