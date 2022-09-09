from ramanchada2 import spectrum
import numpy as np


def test_generate_and_fit():
    lines = {40: 20, 150: 15, 200: 30, 500: 50, 550: 5}
    shift = 50
    spe = spectrum.from_delta_lines(lines).convolve('voigt', sigma=3)
    spe = spe.add_baseline(n_freq=5, amplitude=3, pedestal=0, rng_seed=1111)
    spe = spe.add_poisson_noise(.0002, rng_seed=1111)
    spe = spe.scale_xaxis_fun(lambda x: x - shift)
    candidates = spe.find_peak_groups(prominence=.1, n_sigma_group=0.001, wlen=50, moving_minimum_window=50)

    true_pos = np.array(list(lines.keys()))
    calc_pos = [i for gr in candidates for i in gr.positions]
    fit_peaks = spe.fit_peak_groups(model='Voigt', peak_candidate_groups=candidates)
    fit_pos = fit_peaks.locations

    assert len(true_pos) == len(calc_pos), 'wrong number of peaks found'
    assert np.max(np.abs(true_pos - fit_pos - shift)) < 3, 'fit locations far from truth'
    assert np.max(np.abs(true_pos - calc_pos - shift)) < 5, 'find_peaks locations far from truth'
