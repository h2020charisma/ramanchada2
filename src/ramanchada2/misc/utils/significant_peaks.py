import numpy as np


def calc_weight(pos, amp, locality_factor=10):
    pos = pos - pos.min()
    pos /= pos.max()
    m = np.subtract.outer(pos, pos)
    m = np.abs(m)
    w = amp/((1/np.exp(locality_factor*m)) @ amp)
    return w


def select_peaks(pos, amp, npeaks, **kwargs):
    w = calc_weight(pos, amp, **kwargs)
    idx = np.sort(np.argsort(w)[-npeaks:])
    return idx
