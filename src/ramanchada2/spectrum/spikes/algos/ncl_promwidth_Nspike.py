import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.stats import median_abs_deviation


def metric(y, prominence_threshold=None):
    """
    https://doi.org/10.1016/j.aca.2024.342312
    """
    if prominence_threshold is None:
        prominence_threshold = prominence_threshold = 5*median_abs_deviation((np.diff(y, prepend=[0])))
    peaks, _ = find_peaks(y, prominence=prominence_threshold)
    widths = peak_widths(y, peaks, rel_height=0.5)[0]
    y_merit = np.zeros_like(y)
    for peak, width in zip(peaks, widths):
        y_merit[peak] = width
    return y_merit


def indices(y, width_threshold=None, prominence_threshold=None, width_param_rel=None):
    """
    https://doi.org/10.1016/j.aca.2024.342312
    """
    if width_threshold is None:
        width_threshold = 3
    if prominence_threshold is None:
        prominence_threshold = 5*median_abs_deviation((np.diff(y, prepend=[0])))
    if width_param_rel is None:
        width_param_rel = 0.8
    peaks, _ = find_peaks(y, prominence=prominence_threshold)
    spikes = np.zeros_like(y)
    widths = peak_widths(y, peaks)[0]
    widths_ext_a = peak_widths(y, peaks, rel_height=width_param_rel)[2]
    widths_ext_b = peak_widths(y, peaks, rel_height=width_param_rel)[3]
    for a, width, ext_a, ext_b in zip(range(len(widths)), widths, widths_ext_a, widths_ext_b):
        if width < width_threshold:
            spikes[int(ext_a)-1:int(ext_b)+2] = 1
    return np.where(spikes)[0]
