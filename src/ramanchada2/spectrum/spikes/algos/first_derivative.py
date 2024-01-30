import numpy as np


def metric(intensity):
    """
    This function returns the indices of posible spikes in a spectrum.
    The function calculates the first derivative of a spectrum and identifies the
    indices where te absolute value of this derivative exceeds a specified threshold.

    Parameters:
    intensity (array): An array of intensity values (spectrum).
    threshold (float, optional): A threshold value to identify significant changes in the intensity denoting a spike.
    Defaults to ???.

    Returns:
    array: An array of indices where posible spikes are present.
    """

    # TODO: Include calculation of automatic threshold #

    # Calculate the first derivative (y_merit) of the intensity array.
    # np.diff computes the difference between consecutive elements in the array.
    # 'prepend=[0]' adds a 0 at the beginning to keep the size of the array consistent.
    y_merit = np.abs(np.diff(intensity, prepend=[0]))
    return y_merit


def indices(intensity, threshold=None):
    if threshold is None:
        threshold = 3.5
    return np.where(metric(intensity) > threshold)[0]
