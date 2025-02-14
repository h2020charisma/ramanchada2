from typing import Union

import numpy as np
from pydantic import validate_call
from scipy.stats import median_abs_deviation


def metric(intensity):
    """
    This function returns the indices of posible spikes in a spectrum.
    The function calculates the second derivative (laplacian) in one dimension of a spectrum and identifies the
    indices where te absolute value of this second derivative exceeds a specified threshold.

    Parameters:
    intensity (array): An array of intensity values (spectrum).
    threshold (float, optional): A threshold value to identify significant changes in the intensity denoting a spike.
    Defaults to ???.

    Returns:
    array: An array of indices where posible spikes are present (or where the second derivative of the intensity
    exceeds the threshold).
    """

    # TODO: Include calculation of automatic threshold #

    intensity = intensity/median_abs_deviation(np.diff(intensity))
    y_merit = np.diff(intensity, n=2)
    y_merit = np.abs(np.pad(y_merit, (1, 1), 'constant'))

    y_merit[:2] = 0
    y_merit[-2:] = 0
    return y_merit


@validate_call()
def bool_hot(intensity, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 20.21
    return metric(intensity) > threshold


def indices_(intensity, threshold: Union[None, float] = None):

    # Compare the y_merit (absolute value of the second derivative or laplacian) with the threshold.
    # This creates a boolean array where True (1) indicates a point
    # where the absolute value of the derivative exceeds the threshold (spike).

    if threshold is None:
        threshold = 20.21
    y_merit = metric(intensity)
    spikes_yesno = np.abs(y_merit) > threshold

    # Extract the indices where spikes_yesno is True (i.e., where the merit function exceeds the threshold).
    # np.where returns a tuple, and [0] accesses the first element of the tuple, which contains the indices.
    indices = np.where(spikes_yesno == 1)[0]

    # Return the array of indices.
    return indices
