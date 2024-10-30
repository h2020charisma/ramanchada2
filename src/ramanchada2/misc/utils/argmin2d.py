from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import PositiveInt, validate_call
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def argmin2d(A):
    ymin_idx = np.argmin(A, axis=0)
    xmin_idx = np.argmin(A, axis=1)
    x_idx = np.unique(xmin_idx[xmin_idx[ymin_idx[xmin_idx]] == xmin_idx])
    y_idx = np.unique(ymin_idx[ymin_idx[xmin_idx[ymin_idx]] == ymin_idx])
    matches = np.stack([y_idx, x_idx]).T
    return matches


def find_closest_pairs_idx(x, y):
    outer_dif = np.abs(np.subtract.outer(x, y))
    return argmin2d(outer_dif).T


def find_closest_pairs(x, y):
    x_idx, y_idx = find_closest_pairs_idx(x, y)
    return x[x_idx], y[y_idx]


@validate_call(config=dict(arbitrary_types_allowed=True))
def align(x, y,
          p0: Union[List[float], npt.NDArray] = [0, 1, 0, 0],
          func=lambda x, a0, a1, a2, a3: (a0*np.ones_like(x), a1*x, a2*x**2/1, a3*(x/1000)**3),
          max_iter: PositiveInt = 1000):
    """
    Iteratively finds best match between x and y and evaluates the x scaling parameters.
    min((lambda(x, *p)-y)**2 | *p)
    Finds best parameters *p that minimise L2 distance between scaled x and original y
    """
    if isinstance(p0, list):
        p = np.array(p0)
    else:
        p = p0
    loss = np.inf
    cur_x = x
    for it in range(max_iter):
        cur_x = np.sum(func(x, *p), axis=0)
        x_idx, y_idx = find_closest_pairs_idx(cur_x, y)
        x_match, y_match = x[x_idx], y[y_idx]
        p_bak = p
        obj_mat = np.stack(func(x_match, *np.ones_like(p)), axis=1)
        p, *_ = linalg.lstsq(obj_mat, y_match, cond=1e-8)
        loss_bak = loss
        loss = np.sum((x_match-y_match)**2)/len(x_match)**2
        if np.allclose(p, p_bak):
            break
        if loss > loss_bak:
            pass
            return p_bak
    return p


@validate_call(config=dict(arbitrary_types_allowed=True))
def align_shift(x, y,
                p0: float = 0,
                max_iter: PositiveInt = 1000):
    loss = np.inf
    cur_x = x
    p = p0
    for it in range(max_iter):
        cur_x = x + p
        x_idx, y_idx = find_closest_pairs_idx(cur_x, y)
        x_match, y_match = x[x_idx], y[y_idx]
        p_bak = p
        p = np.mean(y_match-x_match)
        loss_bak = loss
        loss = np.sum((y_match-x_match)**2)
        if np.allclose(p, p_bak):
            break
        if loss > loss_bak:
            return p_bak
    return p


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
    kmeans = KMeans(n_clusters=n_ref if n_ref > n_spe else n_spe, random_state= 68)
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

def cost_function_position(p1, p2, order_weight=1.0 ,priority_weight=1.0):
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
    #distance_cost = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return order_penalty - priority_bonus


def normalize_tuples(tuples):
    second_values = np.array([x[1] for x in tuples])
    min_val, max_val = second_values.min(), second_values.max()
    normalized_values = (second_values - min_val) / (max_val - min_val)
    # Replace the original second dimension with the normalized values
    return [(tuples[i][0], normalized_values[i]) for i in range(len(tuples))]


def cost_matrix_peaks(spectrum_a_dict,spectrum_b_dict,threshold_max_distance=9, cost_func = None):
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

    min_peak_b = np.min(peaks_b)-threshold_max_distance
    max_peak_b = np.max(peaks_b)+threshold_max_distance

    num_peaks_a = len(peaks_a)  
    cost_matrix = np.full((num_peaks_a, num_peaks_b), np.inf)  # Initialize with infinity

    for i in range(num_peaks_a):
        #if peaks_a[i] < min_peak_b or peaks_a[i] > max_peak_b:
        #    continue
        for j in range(num_peaks_b):
            position_cost = abs(peaks_a[i] - peaks_b[j])
            #if position_cost > threshold_max_distance*2:
            #    continue
            cost = cost_func([peaks_a[i],intensities_a_normalized[i]],[peaks_b[j],intensities_b_normalized[j]],priority_weight=2)
             
            cost_matrix[i, j] =  cost
    return cost_matrix

def match_peaks(spectrum_a_dict,spectrum_b_dict,threshold_max_distance=9, df=False , cost_func = None):
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
        Default is 5. 

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
    cost_matrix = cost_matrix_peaks(spectrum_a_dict,spectrum_b_dict,threshold_max_distance=threshold_max_distance*2, cost_func = cost_function if cost_func is None else cost_func)        
    
    # Use the Hungarian algorithm to find the optimal assignment
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

    except Exception as err:
        raise err

    # experiments with dynamic programming
    #print(normalize_tuples(list(spectrum_a_dict.items())))

    #matches, min_cost, _df =  partial_ordered_match(normalize_tuples(list(spectrum_a_dict.items())), normalize_tuples(list(spectrum_b_dict.items())), 
    #           order_weight=1.0, priority_weight=0, cost_func = cost_function_euclidean)
    #print("matches",matches)

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
    #sorted_indices = np.argsort(matched_peaks_a)
    #matched_peaks_a = matched_peaks_a[sorted_indices]
    #matched_peaks_b = matched_peaks_b[sorted_indices]
    #matched_distances = matched_distances[sorted_indices]  

    if df:
        df = pd.DataFrame({
                'spe': matched_peaks_a,
                'reference': matched_peaks_b,
                'distances': matched_distances,
                'intensity_a' : intensity_a,
                'intensity_b' : intensity_b
            })    
    else:
        df = None  
    return (matched_peaks_a,matched_peaks_b,matched_distances,cost_matrix, df)

