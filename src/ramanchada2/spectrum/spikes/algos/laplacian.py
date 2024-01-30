import numpy as np


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

    y_merit = np.diff(intensity, n=2)
    y_merit = np.pad(y_merit, (1, 1), 'constant')


def indices(intensity, threshold=200):

    # Compare the y_merit (absolute value of the second derivative or laplacian) with the threshold.
    # This creates a boolean array where True (1) indicates a point
    # where the absolute value of the derivative exceeds the threshold (spike).

    y_merit = metric(intensity)
    spikes_yesno = np.abs(y_merit) > threshold

    # Extract the indices where spikes_yesno is True (i.e., where the merit function exceeds the threshold).
    # np.where returns a tuple, and [0] accesses the first element of the tuple, which contains the indices.
    indices = np.where(spikes_yesno == 1)[0]

    # Return the array of indices.
    return indices
