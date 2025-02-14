from typing import Union

import numpy as np
from pydantic import validate_call
from scipy.stats import median_abs_deviation


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
    intensity = intensity/median_abs_deviation(np.diff(intensity))
    y_merit = np.abs(np.pad(np.diff(intensity), (1, 0), 'edge'))
    y_merit[:2] = 0
    y_merit[-2:] = 0
    return np.abs(y_merit)


@validate_call()
def bool_hot(intensity, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 21.45
    return metric(intensity) > threshold


@validate_call()
def indices_(intensity, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 21.45
    return np.where(metric(intensity) > threshold)[0]
