import numpy as np
import math

def metric(intensity, K=4, M=2):
    """
    Calculates metrics for spike detection in a Raman spectrum.

    Parameters:
    - intensity (numpy.ndarray): The intensity of the Raman spectrum.
    - coef_adev (int): Coefficient for average deviation.
    - K (int): Order of the Daubechies wavelet used.
    - M (int): Level of wavelet decomposition.

    Returns:
    - residuals (numpy.ndarray): Residuals between the differentiated series and the reconstructed series.
    - adev (float): Average deviation of the signal.
    - coef_adev (float): Coefficient for average deviation.
    """
    # Differentiate the input signal
    diff_series = np.diff(intensity)
    wavelet = 'db' + str(K)

    # Apply wavelet filter transformation and reconstruct the series
    wavelet_transform = pywt.wavedec(diff_series, wavelet, mode='periodic', level=M)
    for i in range(1, M + 1):
        wavelet_transform[i] = np.zeros_like(wavelet_transform[i])
    reconstructed_series = pywt.waverec(wavelet_transform, wavelet, mode='periodic')

    # Calculate residuals
    residuals = np.subtract(diff_series, reconstructed_series[1:])

    # Calculate ADEV
    second_diff = np.diff(diff_series)
    adev = math.sqrt(np.sum(second_diff ** 2) / (2 * (len(second_diff) - 2)))

    return residuals, adev

def indices(intensity, coef_adev=3, K=4, M=2):
    """
    Calculates spike detection indices in a Raman spectrum based on previously calculated metrics.

    Parameters:
    - intensity (numpy.ndarray): The intensity of the Raman spectrum.
    - residuals (numpy.ndarray): Residuals from the calculate_metrics function.
    - adev (float): Average deviation from the calculate_metrics function.
    - coef_adev (float): Coefficient for average deviation.

    Returns:
    - numpy.ndarray: Boolean array indicating spike locations in the spectrum.
    """
    # Initial spike detections based on threshold
    detections = np.abs(metric(intensity)[0]) > coef_adev * metric(intensity)[1]
    additional_detections = True

    # Perform additional detections
    while additional_detections:
        additional_detections = False
        for i in range(1, len(detections) - 1):
            if not detections[i] and (detections[i - 1] or detections[i + 1]):
                new_value = abs(np.diff(intensity)[i]) > coef_adev * metric(intensity)[1]
                detections[i] = new_value
                additional_detections = additional_detections or new_value

    # Preserve algorithm accuracy by marking the first element as False
    detections = np.insert(detections, 0, False)

    return detections
