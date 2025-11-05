import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, Tuple
from numpy.typing import NDArray


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
    x_spe = np.array([], dtype=float)
    x_reference = np.array([], dtype=float)
    x_distance = np.array([], dtype=float)
    clusters = np.array([], dtype=float)

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


def match_peaks_cluster_robust(
    spe_pos_dict: dict[float, float],
    ref: dict[float, float],
    cost_intensity=0.25,
    filter_outliers=True,
    outlier_threshold=3.0,  # in standard deviations
    verbose=False
):
    """Original cluster-based matcher with optional outlier filtering."""
    
    # --- Original cluster matcher ---
    x_spe, x_ref, x_dist, df = match_peaks_cluster(
        spe_pos_dict, ref, _filter_range=True, cost_intensity=cost_intensity
    )

    if len(x_spe) < 2:
        return x_spe, x_ref, x_dist, df

    # --- Outlier filtering ---
    if filter_outliers:
        distances = x_spe - x_ref
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        mask = np.abs(distances - mean_d) <= outlier_threshold * std_d
        x_spe = x_spe[mask]
        x_ref = x_ref[mask]
        x_dist = x_dist[mask]
        if verbose:
            print(f"[Filtered] {np.sum(~mask)} outliers removed, {len(x_spe)} matches remain")

    # Ensure monotonic order
    sort_idx = np.argsort(x_spe)
    x_spe = x_spe[sort_idx]
    x_ref = x_ref[sort_idx]
    x_dist = x_dist[sort_idx]

    return x_spe, x_ref, x_dist, df


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

    # --- Handle default tolerance ---
    if tolerance is None:
        diffs = np.diff(sorted(ref.keys()))
        tolerance = np.min(diffs) if len(diffs) > 0 else 1.0
        print(f"[match_peaks_optimized] tolerance auto-set to {tolerance}")

    # --- Prepare sorted arrays ---
    ref_peaks = np.asarray(sorted(ref.keys()), dtype=float)
    found_peaks = np.asarray(sorted(spe_pos_dict.keys()), dtype=float)
    ref_intensities = np.asarray([ref[k] for k in sorted(ref.keys())], dtype=float)
    found_intensities = np.asarray([spe_pos_dict[k] for k in sorted(spe_pos_dict.keys())], dtype=float)

    # Normalize intensities to [0,1]
    if ref_intensities.max() > 0:
        ref_intensities /= ref_intensities.max()
    if found_intensities.max() > 0:
        found_intensities /= found_intensities.max()

    n_ref, n_found = len(ref_peaks), len(found_peaks)

    # --- Build finite cost matrix ---
    large_cost = 1e6
    cost_matrix = np.full((n_found, n_ref), large_cost)
    for i, f in enumerate(found_peaks):
        for j, r in enumerate(ref_peaks):
            dist = abs(f - r)
            if relative:
                dist /= r
            if dist <= tolerance:
                inten_diff = abs(found_intensities[i] - ref_intensities[j])
                cost_matrix[i, j] = dist + weight_intensity * inten_diff

    # --- Mask out rows/cols with no valid candidates ---
    valid_rows = np.any(cost_matrix < large_cost, axis=1)
    valid_cols = np.any(cost_matrix < large_cost, axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        raise ValueError("No valid matches found within tolerance!")

    cost_sub = cost_matrix[np.ix_(valid_rows, valid_cols)]

    # --- Solve assignment ---
    row_ind_sub, col_ind_sub = linear_sum_assignment(cost_sub)
    row_ind = np.where(valid_rows)[0][row_ind_sub]
    col_ind = np.where(valid_cols)[0][col_ind_sub]

    # --- Filter out invalid high-cost assignments ---
    mask_valid = cost_matrix[row_ind, col_ind] < large_cost
    row_ind, col_ind = row_ind[mask_valid], col_ind[mask_valid]

    matched_spe = found_peaks[row_ind]
    matched_ref = ref_peaks[col_ind]
    distances = matched_ref - matched_spe

    # --- Enforce strict monotonicity ---
    # Sort by reference coordinate
    order = np.argsort(matched_ref)
    matched_spe = matched_spe[order]
    matched_ref = matched_ref[order]
    distances = distances[order]

    # Remove crossings (both spe and ref must increase)
    inc_mask = np.r_[True, (np.diff(matched_spe) > 0) & (np.diff(matched_ref) > 0)]
    matched_spe = matched_spe[inc_mask]
    matched_ref = matched_ref[inc_mask]
    distances = distances[inc_mask]

    # --- Build output DataFrame ---
    df = pd.DataFrame({
        "spe": matched_spe,
        "reference": matched_ref,
        "distance": distances
    })

    return matched_spe, matched_ref, distances, cost_matrix, df


def match_peaks_monotonic_simple(
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
        diffs = np.diff(sorted(ref.keys()))
        tolerance = np.min(diffs) if len(diffs) > 0 else 1.0
        print(f"[match_peaks_monotonic] tolerance auto-set to {tolerance}")
        
    # Sort peaks
    ref_peaks = np.asarray(sorted(ref.keys()), dtype=float)
    found_peaks = np.asarray(sorted(spe_pos_dict.keys()), dtype=float)
    ref_intensities = np.asarray([ref[k] for k in sorted(ref.keys())], dtype=float)
    found_intensities = np.asarray([spe_pos_dict[k] for k in sorted(spe_pos_dict.keys())], dtype=float)

    matched_ref: list[float] = []
    matched_spe: list[float] = []
    distances: list[float] = []

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

    matched_ref_array = np.asarray(matched_ref, dtype=float)
    matched_spe_array = np.asarray(matched_spe, dtype=float)
    distances_array = np.asarray(distances, dtype=float)

    # Compute optional intensity difference (for info)
    ref_int_dict = dict(zip(ref_peaks, ref_intensities))
    spe_int_dict = dict(zip(found_peaks, found_intensities))
    inten_diff = np.array([
        abs(spe_int_dict[s] - ref_int_dict[r]) / (spe_int_dict[s] + ref_int_dict[r] + 1e-9)
        if (s in spe_int_dict and r in ref_int_dict) else np.nan
        for s, r in zip(matched_spe_array, matched_ref_array)
    ])

    df = pd.DataFrame({
        "spe": matched_spe,
        "reference": matched_ref_array,
        "distance": distances_array,
        "intensity_diff": inten_diff
    })

    return matched_spe_array, matched_ref_array, distances_array, df


def match_peaks_monotonic_dynamic_programming(
    spe_pos_dict: Dict[float, float],
    ref: Dict[float, float],
    tolerance: float = 2.0,
    relative: bool = False,
    weight_intensity: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Match peaks monotonically (order-preserving) using dynamic programming.
    Finds the maximal set of matches within tolerance.

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
        Optional weighting for intensity difference (not used in DP, but reported).

    Returns
    -------
    matched_spe : np.ndarray
    matched_ref : np.ndarray
    distances : np.ndarray
    df : pd.DataFrame
    """

    if tolerance is None:
        diffs = np.diff(sorted(ref.keys()))
        tolerance = np.min(diffs) if len(diffs) > 0 else 1.0
        print(f"[match_peaks_monotonic] tolerance auto-set to {tolerance}")    

    ref_peaks = np.array(sorted(ref.keys()), dtype=float)
    spe_peaks = np.array(sorted(spe_pos_dict.keys()), dtype=float)
    ref_intensities = np.array([ref[k] for k in sorted(ref.keys())], dtype=float)
    spe_intensities = np.array([spe_pos_dict[k] for k in sorted(spe_pos_dict.keys())], dtype=float)

    n_ref = len(ref_peaks)
    n_spe = len(spe_peaks)

    # DP table: dp[i,j] = max matches using ref[:i], spe[:j]
    dp = np.zeros((n_ref + 1, n_spe + 1), dtype=int)
    back = np.zeros((n_ref + 1, n_spe + 1), dtype=int)  # 0: none, 1: match, 2: skip ref, 3: skip spe

    # Fill DP table
    for i in range(1, n_ref + 1):
        for j in range(1, n_spe + 1):
            ref_val = ref_peaks[i - 1]
            spe_val = spe_peaks[j - 1]
            tol = tolerance * ref_val if relative else tolerance
            diff = abs(spe_val - ref_val)

            if diff <= tol:
                # Match possible
                dp[i, j] = dp[i - 1, j - 1] + 1
                back[i, j] = 1
            else:
                if dp[i - 1, j] >= dp[i, j - 1]:
                    dp[i, j] = dp[i - 1, j]
                    back[i, j] = 2  # skip ref
                else:
                    dp[i, j] = dp[i, j - 1]
                    back[i, j] = 3  # skip spe

    # Traceback to get matched pairs
    matched_ref: list[float] = []
    matched_spe: list[float] = []

    i, j = n_ref, n_spe
    while i > 0 and j > 0:
        if back[i, j] == 1:
            matched_ref.append(ref_peaks[i - 1])
            matched_spe.append(spe_peaks[j - 1])
            i -= 1
            j -= 1
        elif back[i, j] == 2:
            i -= 1
        else:
            j -= 1

    # Reverse because traceback goes from end
    matched_ref = np.array(matched_ref[::-1], dtype=float)
    matched_spe = np.array(matched_spe[::-1], dtype=float)
    distances = matched_spe - matched_ref

    # Compute intensity differences
    ref_int_dict = dict(zip(ref_peaks, ref_intensities))
    spe_int_dict = dict(zip(spe_peaks, spe_intensities))
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
    matched_spe_array: NDArray[np.float64] = np.array(matched_spe[::-1], dtype=float)
    matched_ref_array: NDArray[np.float64] = np.array(matched_ref[::-1], dtype=float)
    return matched_spe_array, matched_ref_array, distances, df


def match_peaks_monotonic(
    spe_pos_dict: Dict[float, float],
    ref: Dict[float, float],
    tolerance: float = 9,
    relative: bool = False,
    weight_intensity: float = 0.5,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Monotonic peak matching: returns maximal subset of matches.
    Peaks that cannot be matched are skipped.
    """

    ref_peaks = np.array(sorted(ref.keys()), dtype=float)
    spe_peaks = np.array(sorted(spe_pos_dict.keys()), dtype=float)
    ref_int = np.array([ref[k] for k in sorted(ref.keys())], dtype=float)
    spe_int = np.array([spe_pos_dict[k] for k in sorted(spe_pos_dict.keys())], dtype=float)

    if len(ref_peaks) == 0 or len(spe_peaks) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), pd.DataFrame()

    # default tolerance
    if tolerance is None:
        diffs = np.diff(ref_peaks)
        tolerance = 0.5 * np.median(diffs) if len(diffs) > 0 else 10.0
        if debug:
            print(f"[match_peaks_monotonic_partial] tolerance auto-set to {tolerance:.3f}")

    matched_ref, matched_spe, distances = _dp_match_partial_core(
        ref_peaks, ref_int, spe_peaks, spe_int,
        tolerance=tolerance,
        relative=relative,
        weight_intensity=weight_intensity,
        debug=debug,
    )

    inten_diff = np.array([
        abs(spe_pos_dict[s] - ref[r]) / (spe_pos_dict[s] + ref[r] + 1e-9)
        for s, r in zip(matched_spe, matched_ref)
    ])

    df = pd.DataFrame({
        "spe": matched_spe,
        "reference": matched_ref,
        "distance": distances,
        "intensity_diff": inten_diff
    })

    return matched_spe, matched_ref, distances, df


def _dp_match_partial_core(ref_peaks, ref_int, spe_peaks, spe_int,
                           tolerance, relative, weight_intensity, debug=False):
    """DP core allowing skipping peaks: returns maximal monotonic subset."""

    n_ref, n_spe = len(ref_peaks), len(spe_peaks)
    # length of longest match ending at i,j
    dp_len = np.zeros((n_ref + 1, n_spe + 1), dtype=int)
    # traceback
    back = np.zeros((n_ref + 1, n_spe + 1, 2), dtype=int)  # store previous i,j

    for i in range(1, n_ref + 1):
        for j in range(1, n_spe + 1):
            ref_val, spe_val = ref_peaks[i - 1], spe_peaks[j - 1]
            tol = tolerance * ref_val if relative else tolerance

            if abs(spe_val - ref_val) <= tol:
                # match: length = 1 + dp from previous
                dp_len[i, j] = dp_len[i - 1, j - 1] + 1
                back[i, j] = [i - 1, j - 1]
            else:
                # skip either ref or spe: take max length
                if dp_len[i - 1, j] >= dp_len[i, j - 1]:
                    dp_len[i, j] = dp_len[i - 1, j]
                    back[i, j] = [i - 1, j]
                else:
                    dp_len[i, j] = dp_len[i, j - 1]
                    back[i, j] = [i, j - 1]

        if debug:
            candidates = np.sum(np.abs(spe_peaks - ref_val) <= tol)
            print(f"Row {i}/{n_ref} filled. candidates within tol: {candidates}, longest subset={np.max(dp_len[i,:])}")

    # Traceback: maximal subset
    i, j = n_ref, n_spe
    matched_ref, matched_spe = [], []
    while i > 0 and j > 0:
        prev_i, prev_j = back[i, j]
        if prev_i == i - 1 and prev_j == j - 1:
            # matched
            matched_ref.append(ref_peaks[i - 1])
            matched_spe.append(spe_peaks[j - 1])
        i, j = prev_i, prev_j

    matched_ref = np.array(matched_ref[::-1])
    matched_spe = np.array(matched_spe[::-1])
    distances = matched_spe - matched_ref
    if debug:
        print(f"[DP partial] Traceback complete. {len(matched_ref)} matches, mean distance {np.mean(distances) if len(distances)>0 else np.nan:.3f}")

    return matched_ref, matched_spe, distances
