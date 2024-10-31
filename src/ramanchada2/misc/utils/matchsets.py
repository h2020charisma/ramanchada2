import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
import numpy as np


def match_peaks_cluster(spe_pos_dict, ref):
    # Min-Max normalize the reference values
    min_value = min(ref.values())
    max_value = max(ref.values())
    if len(ref.keys()) > 1:
        normalized_ref = {key: (value - min_value) / (max_value - min_value) for key, value in ref.items()}
    else:
        normalized_ref = ref

    min_value_spe = min(spe_pos_dict.values())
    max_value_spe = max(spe_pos_dict.values())
    # Min-Max normalize the spe_pos_dict
    if len(spe_pos_dict.keys()) > 1:
        normalized_spe = {
                key: (value - min_value_spe) / (max_value_spe - min_value_spe) for key, value in spe_pos_dict.items()
                }
    else:
        normalized_spe = spe_pos_dict
    data_list = [
        {'Wavelength': key, 'Intensity': value, 'Source': 'spe'} for key, value in normalized_spe.items()
    ] + [
        {'Wavelength': key, 'Intensity': value, 'Source': 'reference'} for key, value in normalized_ref.items()
    ]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    feature_matrix = df[['Wavelength', 'Intensity']].to_numpy()

    n_ref = len(ref.keys())
    n_spe = len(spe_pos_dict.keys())
    kmeans = KMeans(n_clusters=n_ref if n_ref > n_spe else n_spe, random_state=68)
    kmeans.fit(feature_matrix)
    labels = kmeans.labels_
    # Extract cluster labels, x values, and y values
    df['Cluster'] = labels
    grouped = df.groupby('Cluster')
    x_spe = np.array([])
    x_reference = np.array([])
    x_distance = np.array([])
    clusters = np.array([])
    # Iterate through each group
    for cluster, group in grouped:
        # Get the unique sources within the group
        unique_sources = group['Source'].unique()
        if 'reference' in unique_sources and 'spe' in unique_sources:
            # Pivot the DataFrame to create the desired structure
            for w_spe in group.loc[group["Source"] == "spe"]["Wavelength"].values:
                x = None
                r = None
                e_min = None
                for w_ref in group.loc[group["Source"] == "reference"]["Wavelength"].values:
                    e = euclidean_distances(w_spe.reshape(-1, 1), w_ref.reshape(-1, 1))[0][0]
                    if (e_min is None) or (e < e_min):
                        x = w_spe
                        r = w_ref
                        e_min = e
            x_spe = np.append(x_spe, x)
            x_reference = np.append(x_reference, r)
            x_distance = np.append(x_distance, e_min)
            clusters = np.append(clusters, cluster)
    sort_indices = np.argsort(x_spe)
    return (x_spe[sort_indices], x_reference[sort_indices], x_distance[sort_indices], df)


def cost_function_position(p1, p2, order_weight=1.0, priority_weight=1.0):
    order_penalty = order_weight * abs(p1[0] - p2[0])
    return order_penalty


def cost_function(p1, p2, order_weight=1.0, priority_weight=.1):
    """
    Modified cost function with an order preservation penalty and priority weighting.
    - `order_weight` increases penalty for large differences in the x-axis values.
    - `priority_weight` decreases the cost for higher values in the y-axis for set_b points.
    """
    order_penalty = order_weight * abs(p1[0] - p2[0])
    priority_bonus = priority_weight * p2[1]  # Rewards points in set_b with higher second dimension values
    return order_penalty - priority_bonus


def normalize_tuples(tuples):
    second_values = np.array([x[1] for x in tuples])
    min_val, max_val = second_values.min(), second_values.max()
    normalized_values = (second_values - min_val) / (max_val - min_val)
    # Replace the original second dimension with the normalized values
    return [(tuples[i][0], normalized_values[i]) for i in range(len(tuples))]


def cost_matrix_peaks(spectrum_a_dict, spectrum_b_dict, threshold_max_distance=9, cost_func=None):
    if cost_func is None:
        cost_func = cost_function_position
    peaks_a = np.array(list(spectrum_a_dict.keys()))
    intensities_a = np.array(list(spectrum_a_dict.values()))
    peaks_b = np.array(list(spectrum_b_dict.keys()))
    intensities_b = np.array(list(spectrum_b_dict.values()))

    num_peaks_b = len(peaks_b)  # Number of reference peaks to match

    # Normalize intensities using min-max normalization
    def normalize_intensities(intensities):
        min_intensity = np.min(intensities)
        max_intensity = np.max(intensities)
        return (intensities - min_intensity) / (max_intensity - min_intensity)

    intensities_a_normalized = normalize_intensities(intensities_a)
    intensities_b_normalized = normalize_intensities(intensities_b)

    num_peaks_a = len(peaks_a)
    cost_matrix = np.full((num_peaks_a, num_peaks_b), np.inf)  # Initialize with infinity

    for i in range(num_peaks_a):
        for j in range(num_peaks_b):
            cost = cost_func([peaks_a[i], intensities_a_normalized[i]], 
                             [peaks_b[j], intensities_b_normalized[j]], 
                             priority_weight=1
                             )
            cost_matrix[i, j] = cost
    return cost_matrix


def match_peaks(spectrum_a_dict, spectrum_b_dict, threshold_max_distance=9, df=False, cost_func=None):
    """
    Match peaks between two spectra based on their positions and intensities.

    Uses scipy linear_sum_assignment to match peaks based on cost function
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

    Parameters:
    ----------
    spectrum_a_dict : dict
        A dictionary representing the first spectrum, where keys are peak
        positions (float) and values are peak intensities (float).

    spectrum_b_dict : dict
        A dictionary representing the second spectrum, where keys are peak
        positions (float) and values are peak intensities (float).

    threshold_max_distance : float, optional
        The maximum allowed distance for two peaks to be considered a match.
        Default is 8.

    df : bool, optional
        If True, return a DataFrame with matched peaks and their respective
        intensities; if False, return None

    Returns:
    -------
    matched_peaks : (matched_peaks_a,matched_peaks_b,matched_distances, df)

    Examples:
    ---------
    >>> spectrum_a = {100: 10, 105: 20, 110: 15}
    >>> spectrum_b = {102: 12, 106: 22, 111: 16}
    >>> match_peaks(spectrum_a, spectrum_b)

    """
    cost_matrix = cost_matrix_peaks(spectrum_a_dict, spectrum_b_dict, 
                                    threshold_max_distance=threshold_max_distance, 
                                    cost_func=cost_function if cost_func is None else cost_func
                                    )

    # Use the Hungarian algorithm to find the optimal assignment
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

    except Exception as err:
        raise err

    # Prepare matched peaks and distances
    # I am sure this could be done in a more efficient way
    matched_peaks_a = []
    matched_peaks_b = []
    matched_distances = []
    intensity_a = []
    intensity_b = []

    peaks_a = np.array(list(spectrum_a_dict.keys()))
    intensities_a = np.array(list(spectrum_a_dict.values()))
    peaks_b = np.array(list(spectrum_b_dict.keys()))
    intensities_b = np.array(list(spectrum_b_dict.values()))

    last_matched_reference = -np.inf
    last_matched_cost = np.inf
    for i in range(len(row_ind)):
        cost = cost_matrix[row_ind[i], col_ind[i]]
        if abs(peaks_a[row_ind[i]]-peaks_b[col_ind[i]]) >= threshold_max_distance:
            continue
        if cost < np.inf:  # Only consider valid pairs
            current_reference = peaks_b[col_ind[i]]
            if current_reference >= last_matched_reference:
                matched_peaks_a.append(peaks_a[row_ind[i]])
                matched_peaks_b.append(current_reference)
                matched_distances.append(cost)
                last_matched_reference = current_reference
                last_matched_cost = cost
                intensity_a.append(intensities_a[row_ind[i]])
                intensity_b.append(intensities_b[col_ind[i]])
            elif last_matched_cost > cost:
                matched_peaks_a[-1] = peaks_a[row_ind[i]]
                matched_peaks_b[-1] = current_reference
                matched_distances[-1] = cost
                intensity_a[-1] = intensities_a[row_ind[i]]
                intensity_b[-1] = intensities_b[col_ind[i]]
                last_matched_cost = cost

    matched_peaks_a = np.array(matched_peaks_a)
    matched_peaks_b = np.array(matched_peaks_b)
    matched_distances = np.array(matched_distances)

    # Sort matched peaks by peaks_a
    # linear_sum_assignment shall give the row_ind sorted
    # sorted_indices = np.argsort(matched_peaks_a)
    # matched_peaks_a = matched_peaks_a[sorted_indices]
    # matched_peaks_b = matched_peaks_b[sorted_indices]
    # matched_distances = matched_distances[sorted_indices]

    if df:
        df = pd.DataFrame({
                'spe': matched_peaks_a,
                'reference': matched_peaks_b,
                'distances': matched_distances,
                'intensity_a': intensity_a,
                'intensity_b': intensity_b
            })
    else:
        df = None
    return (matched_peaks_a, matched_peaks_b, matched_distances, cost_matrix, df)
