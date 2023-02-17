#!/usr/bin/env python3

import numpy as np
import ramanchada2 as rc2


def test_recover_spikes():
    spe = rc2.spectrum.from_delta_lines(deltas={100: 100, 500: 400, 3000: 300}).convolve('gaussian', sigma=2)
    spe = spe.add_baseline(n_freq=30, amplitude=2, pedestal=30)
    spe = spe.add_poisson_noise(.1)
    spe_bad = spe.__copy__()
    y = spe_bad.y[:]
    y[300] = 0
    y[1000] = 70
    y[1500] = 55
    spe_bad.y = y
    spe_reco = spe_bad.recover_spikes(10)
    assert np.all(spe_bad.x == spe_reco.x), "x axes are expected to be the same"
    modified_bins = set(np.argwhere(spe_bad.y != spe_reco.y).reshape(-1))
    assert modified_bins == {300, 1000, 1500}, f'modifications found in {modified_bins} expected [300, 1000, 1500]'
