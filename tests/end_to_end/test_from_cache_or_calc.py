import os
import tempfile

import numpy as np

import ramanchada2 as rc2
from ramanchada2.misc.types.fit_peaks_result import FitPeaksResult


def test_from_cache_or_calc():
    with tempfile.TemporaryDirectory() as tmpdir:
        steps = [{'proc': 'from_delta_lines',
                  'args': [],
                  'kwargs': {'nbins': 3000, 'deltas': {200: 100, 600: 50, 1000: 150, 1500: 70}}},
                 {'proc': 'normalize', 'args': [], 'kwargs': {}},
                 {'proc': 'add_gaussian_noise_drift', 'args': [], 'kwargs': {'sigma': 1, 'coef': .1}}]

        cachefile = None
        spe1 = rc2.spectrum.from_cache_or_calc(cachefile=cachefile, required_steps=steps)
        spe2 = rc2.spectrum.from_cache_or_calc(cachefile=cachefile, required_steps=steps)
        assert not np.allclose(spe1.y, spe2.y)  # spe2 is distict from spe1. spe2 not coming from cache

        cachefile = ''
        spe1 = rc2.spectrum.from_cache_or_calc(cachefile=cachefile, required_steps=steps)
        spe2 = rc2.spectrum.from_cache_or_calc(cachefile=cachefile, required_steps=steps)
        assert not np.allclose(spe1.y, spe2.y)  # spe2 is distict from spe1. spe2 not coming from cache

        cachefile = os.path.join(tmpdir, 'test.cha')
        spe1 = rc2.spectrum.from_cache_or_calc(cachefile=cachefile, required_steps=steps)
        spe2 = rc2.spectrum.from_cache_or_calc(cachefile=cachefile, required_steps=steps)
        assert np.allclose(spe1.y, spe2.y)  # spe2 is equal to spe1. spe2 is coming from cache

        spe_deltas = rc2.spectrum.from_delta_lines(deltas={200: 100, 600: 50, 1000: 150, 1500: 70},
                                                   nbins=3000,
                                                   cachefile=cachefile)
        spe_gaus = spe_deltas.convolve('gaussian', sigma=20).normalize()
        spe_baseline = spe_gaus.add_baseline(n_freq=20, pedestal=.02, amplitude=.15)
        spe_noise = spe_baseline.add_gaussian_noise(sigma=.05).normalize()
        spe_cand = spe_noise.find_peak_multipeak_filter(prominence=.3, width=25)
        spe_fit = spe_cand.fit_peaks_filter(profile='Gaussian')

        assert len(spe_cand.result) == 4
        assert len(spe_fit.result) == 4

        steps = spe_fit.applied_processings_dict()

        spe_fit_cache = rc2.spectrum.from_cache_or_calc(cachefile=cachefile, required_steps=steps)

        assert np.allclose(spe_fit.y, spe_fit_cache.y)  # make sure spe_fit_cache is coming from cache

        assert FitPeaksResult.loads(spe_fit.result).to_dataframe_peaks().equals(
            FitPeaksResult.loads(spe_fit_cache.result).to_dataframe_peaks())
