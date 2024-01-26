import numpy as np


def mod_zeta_scores(ys, threshold=3.5):
    """
    This approach is based in the publication by
    Whitaker, Darren A., and Kevin Hayes: "A simple algorithm for despiking Raman spectra."
    Chemometrics and Intelligent Laboratory Systems 179 (2018): 82-84.

    This function returns the indices of posible spikes in a spectrum.
    The function calculates the modified z scores of a spectrum and identifies the
    indices where absolute value of the value of these z scores exceeds a specified threshold.

    Parameters:
    intensity (array): An array of intensity values (spectrum).
    threshold (float, optional): A threshold value to identify significant changes in the intensity denoting a spike.
    Defaults to ???.

    Returns:
    array: An array of indices where posible spikes are present (or where the second derivative of the intensity
    exceeds the threshold).
    """

    # two ways to calculate the derivative: Like this, and with np.diff. I saw before that the np.diff gave trouble
    # when working with spectra (matrix)
    dist = 0
    ys_diff = []
    for i in np.arange(len(ys)-1):
        dist = ys[i+1] - ys[i]
        ys_diff.append(dist)

    # Luego calculamos la mediana y el MAD de la serie diferenciada
    median_y = np.median(ys_diff)
    # median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys_diff])
    median_absolute_deviation_y = np.median(np.abs(ys_diff - median_y))

    # Calculamos el valor zeta modificado
    # modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ys_diff]
    modified_z_scores = 0.6745 * (ys_diff - median_y) / median_absolute_deviation_y

    y_merit = np.insert(np.abs(np.array(modified_z_scores)), 0, 0)

    # Alternatively and much more elegant:
    # median_y = np.median(ys_diff)
    # median_absolute_deviation_y =  np.median(np.abs(ys_diff - median_y))
    # modified_z_scores = 0.6745 * (ys_diff - median_y) / median_absolute_deviation_y
    # y_merit = np.abs(modified_z_scores)

    # ################ TO BE DONE #################
    #  Include calculation of automatic threshold #
    #  Threshold = 3.5 recommended by the authors #
    # ############################################

    # Compare the y_merit (absolute value of the second derivative or laplacian) with the threshold.
    # This creates a boolean array where True (1) indicates a point
    # where the absolute value of the derivative exceeds the threshold (spike).
    spikes_yesno = y_merit > threshold

    # Extract the indices where spikes_yesno is True (i.e., where the merit function exceeds the threshold).
    # np.where returns a tuple, and [0] accesses the first element of the tuple, which contains the indices.
    indices = np.where(spikes_yesno == 1)[0]

    # Return the array of indices.
    return indices
