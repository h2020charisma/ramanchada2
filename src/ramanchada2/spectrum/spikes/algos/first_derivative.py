import numpy as np


def metric(intensity):
    return np.diff(intensity, prepend=[0])


def indices(s, threshold=None):
    ...
