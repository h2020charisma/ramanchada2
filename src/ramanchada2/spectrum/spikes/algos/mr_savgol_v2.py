import numpy as np
import math
from scipy.signal import savgol_filter


def metric(intensity, window_length=5, polyorder=2):
    """
    Based on https://doi.org/10.1366/14-07834
    Calculates metrics for spike detection in a Raman spectrum using Savitzky-Golay filter
    for smoothing instead of wavelet transformation.

    Parameters:
    - intensity (numpy.ndarray): The intensity of the Raman spectrum.
    - window_length (int): The length of the filter window (i.e., the number of coefficients).
    window_length must be a positive odd integer.
    - polyorder (int): The order of the polynomial used to fit the samples. polyorder must be less
                       than window_length.

    Returns:
    - residuals (numpy.ndarray): Residuals between the differentiated series and the smoothed series.
    """
    # Smooth the input signal
    smoothed = savgol_filter(intensity, window_length, polyorder)

    # Differentiate the input signal
    diff_series = np.diff(intensity)

    # Calculate residuals between the differentiated series and the smoothed series
    residuals = np.subtract(diff_series, np.diff(smoothed))
    return residuals


def indices(intensity, coef_adev=3, window_length=5, polyorder=2):
    """
    Calculates spike detection indices in a Raman spectrum based on previously calculated metrics.

    Parameters:
    - intensity (numpy.ndarray): The intensity of the Raman spectrum.
    - coef_adev (float): Coefficient for average deviation.
    - window_length (int): The length of the filter window for Savitzky-Golay filter.
    - polyorder (int): The order of the polynomial for Savitzky-Golay filter.

    Returns:
    - numpy.ndarray: Boolean array indicating spike locations in the spectrum.
    """
    # Calculate ADEV using second difference of the intensity
    second_diff = np.diff(intensity, n=2)
    adev = math.sqrt(np.sum(second_diff ** 2) / (2 * (len(second_diff) - 2)))

    # Calculate initial spike detections based on threshold
    detections = np.abs(metric(intensity, window_length, polyorder)) > coef_adev * adev
    additional_detections = True

    # Perform additional detections
    while additional_detections:
        additional_detections = False
        for i in range(1, len(detections) - 1):
            if not detections[i] and (detections[i - 1] or detections[i + 1]):
                new_value = abs(np.diff(intensity, n=1)[i-1]) > coef_adev * adev
                detections[i] = new_value
                additional_detections = additional_detections or new_value

    # Mark the first element as False for algorithm accuracy
    detections = np.insert(detections, 0, False)

    return detections
