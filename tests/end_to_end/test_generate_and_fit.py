from ramanchada2 import spectrum
import numpy as np


def test_generate_and_fit():
    lines = {40: 20, 150: 15, 200: 30, 500: 50, 550: 5}
    spe = spectrum.from_delta_lines(lines).convolve('voigt', sigma=3)
    spe = spe.add_baseline(bandwidth=5, amplitude=3, pedestal=0, rng_seed=1111)
    spe = spe.add_poisson_noise(.0002, rng_seed=1111)
    spe = spe.scale_xaxis_fun(lambda x: x-50)
    spe = spe - spe.moving_minimum(50)
    spe = spe.normalize('minmax')
    found_peaks = spe.find_peaks(prominence=.1)

    true_pos = np.array(list(lines.keys()))
    calc_pos = found_peaks.peaks
    fit_peaks = spe.fit_peaks(model='Voigt', peak_candidates=found_peaks)
    fit_pos = fit_peaks.locations

    assert len(true_pos) == len(calc_pos), 'wrong number of peaks found'
    assert np.max(np.abs(true_pos - fit_pos - 50)) < 3, 'fit locations far from truth'
    assert np.max(np.abs(true_pos - calc_pos)) < 5, 'find_peaks locations far from truth'
