import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, List, Tuple


def match_peaks_cluster(
    spe_pos_dict: Dict[float, float],
    ref: Dict[float, float],
    _filter_range=True,
    cost_intensity=0.25,
):
    wl_label = "Wavelength"
    intensity_label = "Intensity"
    source_label = "Source"

    # Min-Max normalize the reference values
    min_value = min(ref.values())
    max_value = max(ref.values())
    if len(ref.keys()) > 1:
        normalized_ref = {
            key: (value - min_value) / (max_value - min_value)
            for key, value in ref.items()
        }
    else:
        normalized_ref = ref

    min_value_spe = min(spe_pos_dict.values())
    max_value_spe = max(spe_pos_dict.values())
    # Min-Max normalize the spe_pos_dict
    if len(spe_pos_dict.keys()) > 1:
        normalized_spe = {
            key: (value - min_value_spe) / (max_value_spe - min_value_spe)
            for key, value in spe_pos_dict.items()
        }
    else:
        normalized_spe = spe_pos_dict
    data_list = [
        {wl_label: key, intensity_label: value, source_label: "spe"}
        for key, value in normalized_spe.items()
    ] + [
        {wl_label: key, intensity_label: value, source_label: "reference"}
        for key, value in normalized_ref.items()
    ]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    df["intensity4cluster"] = df[intensity_label] * cost_intensity

    if _filter_range:
        _tollerance = 50
        left_limit = max(min(ref.keys()), min(spe_pos_dict.keys())) - _tollerance
        right_limit = min(max(ref.keys()), max(spe_pos_dict.keys())) + _tollerance
        df = df.loc[(df[wl_label] <= right_limit) & (df[wl_label] >= left_limit)]
        n_spe = len(df[df[source_label] == "spe"])
        n_ref = len(df[df[source_label] == "reference"])
    else:
        n_ref = len(ref.keys())
        n_spe = len(spe_pos_dict.keys())

    feature_matrix = df[[wl_label, "intensity4cluster"]].to_numpy()

    n_ref = len(ref.keys())
    n_spe = len(spe_pos_dict.keys())
    kmeans = KMeans(n_clusters=n_ref if n_ref > n_spe else n_spe, random_state=68)
    kmeans.fit(feature_matrix)
    labels = kmeans.labels_
    # Extract cluster labels, x values, and y values
    df["Cluster"] = labels
    grouped = df.groupby("Cluster")
    x_spe = np.array([])
    x_reference = np.array([])
    x_distance = np.array([])
    clusters = np.array([])

    # Iterate through each group
    for cluster, group in grouped:
        # Get the unique sources within the group
        unique_sources = group["Source"].unique()
        if "reference" in unique_sources and "spe" in unique_sources:
            x = None
            r = None
            e_min = None
            for _, row_spe in group.loc[group[source_label] == "spe"].iterrows():
                w_spe = row_spe[wl_label]
                for _, row_ref in group.loc[
                    group[source_label] == "reference"
                ].iterrows():
                    w_ref = row_ref[wl_label]
                    e = (w_spe - w_ref) ** 2 + (
                        row_spe[intensity_label] - row_ref[intensity_label]
                    ) ** 2
                    if (e_min is None) or (e < e_min):
                        x = w_spe
                        r = w_ref
                        e_min = e
            if x is not None and r is not None and e_min is not None:
                x_spe = np.append(x_spe, x)
                x_reference = np.append(x_reference, r)
                x_distance = np.append(x_distance, e_min)
                clusters = np.append(clusters, cluster)
    sort_indices = np.argsort(x_spe)
    return (
        x_spe[sort_indices],
        x_reference[sort_indices],
        x_distance[sort_indices],
        df,
    )


def match_peaks_optimized(
    spe_pos_dict: Dict[float, float],
    ref: Dict[float, float],
    tolerance: float = 2.0,   # e.g. pixels
    relative: bool = False,
    weight_intensity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Match found peaks (spe_pos_dict) to reference peaks (ref) using the Hungarian algorithm.
    Ensures one-to-one matching, tolerance filtering, and strictly increasing results.
    """
    if tolerance is None:
        #tolerance = 0.01 * np.mean(np.diff(sorted(ref.keys())))
        tolerance = np.min(np.diff(sorted(ref.keys())))

    # Sort peaks by position
    ref_peaks = np.asarray(sorted(ref.keys()), dtype=float)
    found_peaks = np.asarray(sorted(spe_pos_dict.keys()), dtype=float)
    ref_intensities = np.asarray([ref[k] for k in sorted(ref.keys())], dtype=float)
    found_intensities = np.asarray([spe_pos_dict[k] for k in sorted(spe_pos_dict.keys())], dtype=float)

    # Normalize intensities
    if ref_intensities.max() > 0:
        ref_intensities /= ref_intensities.max()
    if found_intensities.max() > 0:
        found_intensities /= found_intensities.max()

    n_ref, n_found = len(ref_peaks), len(found_peaks)

    # Build cost matrix
    cost_matrix = np.full((n_found, n_ref), np.inf)
    for i, f in enumerate(found_peaks):
        for j, r in enumerate(ref_peaks):
            dist = abs(f - r)
            if relative:
                dist /= r
            if dist <= tolerance:
                inten_diff = abs(found_intensities[i] - ref_intensities[j])
                cost_matrix[i, j] = dist + weight_intensity * inten_diff

    # --- Handle infeasible matrix ---
    valid_rows = np.any(np.isfinite(cost_matrix), axis=1)
    valid_cols = np.any(np.isfinite(cost_matrix), axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        raise ValueError("No valid matches found within tolerance!")

    cost_sub = cost_matrix[np.ix_(valid_rows, valid_cols)]

    # Solve assignment
    row_ind_sub, col_ind_sub = linear_sum_assignment(cost_sub)

    # Map back to original indices
    row_ind = np.where(valid_rows)[0][row_ind_sub]
    col_ind = np.where(valid_cols)[0][col_ind_sub]

    # Filter valid
    valid_mask = np.isfinite(cost_matrix[row_ind, col_ind])
    row_ind, col_ind = row_ind[valid_mask], col_ind[valid_mask]

    matched_spe = found_peaks[row_ind]
    matched_ref = ref_peaks[col_ind]
    distances = matched_ref - matched_spe

    # --- Enforce strict monotonicity ---
    order = np.argsort(matched_ref)
    matched_spe = matched_spe[order]
    matched_ref = matched_ref[order]
    distances = distances[order]

    # Remove non-increasing pairs
    inc_mask = np.r_[True, np.diff(matched_spe) > 0]
    matched_spe = matched_spe[inc_mask]
    matched_ref = matched_ref[inc_mask]
    distances = distances[inc_mask]

    df = pd.DataFrame({
        "spe": matched_spe,
        "reference": matched_ref,
        "distance": distances
    })

    return matched_spe, matched_ref, distances, cost_matrix, df


def match_peaks_monotonic(
    spe_pos_dict: Dict[float, float],
    ref: Dict[float, float],
    tolerance: float = 2.0,   # in pixels or nm
    relative: bool = False,
    weight_intensity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Match peaks monotonically (order-preserving, one-to-one) ensuring
    |ref - spe| < tolerance. Returns only valid monotonic matches.

    Parameters
    ----------
    spe_pos_dict : Dict[float, float]
        Found peaks {position: intensity}.
    ref : Dict[float, float]
        Reference peaks {position: intensity}.
    tolerance : float
        Absolute or relative tolerance for matching.
    relative : bool
        If True, interpret tolerance as fraction of reference position.
    weight_intensity : float
        Optional weighting for intensity difference.

    Returns
    -------
    matched_spe : np.ndarray
    matched_ref : np.ndarray
    distances : np.ndarray
    df : pd.DataFrame
    """

    if tolerance is None:
        #tolerance = 0.01 * np.mean(np.diff(sorted(ref.keys())))
        tolerance = np.min(np.diff(sorted(ref.keys())))
        
    # Sort peaks
    ref_peaks = np.asarray(sorted(ref.keys()), dtype=float)
    found_peaks = np.asarray(sorted(spe_pos_dict.keys()), dtype=float)
    ref_intensities = np.asarray([ref[k] for k in sorted(ref.keys())], dtype=float)
    found_intensities = np.asarray([spe_pos_dict[k] for k in sorted(spe_pos_dict.keys())], dtype=float)

    matched_ref = []
    matched_spe = []
    distances = []

    i = j = 0
    while i < len(ref_peaks) and j < len(found_peaks):
        ref_val = ref_peaks[i]
        spe_val = found_peaks[j]

        # Compute allowed tolerance
        tol = tolerance * ref_val if relative else tolerance
        diff = spe_val - ref_val

        # Within tolerance
        if abs(diff) <= tol:
            matched_ref.append(ref_val)
            matched_spe.append(spe_val)
            distances.append(diff)
            i += 1
            j += 1
        elif spe_val < ref_val:
            # found peak too low → advance found index
            j += 1
        else:
            # reference too low → advance reference index
            i += 1

    matched_ref = np.asarray(matched_ref)
    matched_spe = np.asarray(matched_spe)
    distances = np.asarray(distances)

    # Compute optional intensity difference (for info)
    ref_int_dict = dict(zip(ref_peaks, ref_intensities))
    spe_int_dict = dict(zip(found_peaks, found_intensities))
    inten_diff = np.array([
        abs(spe_int_dict[s] - ref_int_dict[r]) / (spe_int_dict[s] + ref_int_dict[r] + 1e-9)
        if (s in spe_int_dict and r in ref_int_dict) else np.nan
        for s, r in zip(matched_spe, matched_ref)
    ])

    df = pd.DataFrame({
        "spe": matched_spe,
        "reference": matched_ref,
        "distance": distances,
        "intensity_diff": inten_diff
    })

    return matched_spe, matched_ref, distances, df
