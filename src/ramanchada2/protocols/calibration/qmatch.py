from scipy.interpolate import PchipInterpolator

import numpy as np

def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    a0, a1 = arr.min(), arr.max()
    return (arr - a0) / (a1 - a0), a0, a1

def denormalize(arr_n, a0, a1):
    return arr_n * (a1 - a0) + a0


def quantile_map(x, y, q=(0.05, 0.5, 0.95)):
    """
    Build coarse linear map using quantiles.
    """
    qx = np.quantile(x, q)
    qy = np.quantile(y, q)
    a = np.polyfit(qx, qy, deg=1)
    return np.poly1d(a)


def deduplicate_points(x, y, min_sep=1e-10):
    """
    When x values are too close, average their y values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Sort
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    # Group nearby points
    groups = []
    current_group_x = [x[0]]
    current_group_y = [y[0]]
    
    for i in range(1, len(x)):
        if x[i] - x[i-1] < min_sep:
            # Same group
            current_group_x.append(x[i])
            current_group_y.append(y[i])
        else:
            # New group - save old one
            groups.append((np.mean(current_group_x), np.mean(current_group_y)))
            current_group_x = [x[i]]
            current_group_y = [y[i]]
    
    # Don't forget last group
    groups.append((np.mean(current_group_x), np.mean(current_group_y)))
    
    x_out, y_out = zip(*groups)
    return np.array(x_out), np.array(y_out)


def robust_pchip_fit(x, y, max_iter=5, sigma=0.02):
    """
    Iterative robust monotone fit.
    """
    mask = np.ones(len(x), dtype=bool)

    for _ in range(max_iter):
        spline = PchipInterpolator(x[mask], y[mask])
        resid = y - spline(x)
        mad = np.median(np.abs(resid))
        mask = np.abs(resid) < sigma * max(mad, 1e-6)
        #print(mad)

    return PchipInterpolator(x[mask], y[mask]), mask


def invert_monotone(x, y):
    """Build inverse, keeping only monotonic points."""
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    
    # Keep only strictly increasing y
    keep = [0]
    for i in range(1, len(y)):
        if y[i] > y[keep[-1]] + 1e-10:
            keep.append(i)
    
    if len(keep) < 2:
        raise ValueError(f"Only {len(keep)} monotonic points - calibration failed")
    
    return PchipInterpolator(y[keep], x[keep])


def collapse_candidates_pre_fit(x_n, y_n, f0):
    """
    Enforce one-to-one mapping BEFORE PCHIP.
    For each x, keep the y closest to the coarse model f0(x).
    """
    best = {}

    for xi, yi in zip(x_n, y_n):
        err = abs(yi - f0(xi))
        if xi not in best or err < best[xi][1]:
            best[xi] = (yi, err)

    x_out = np.array(list(best.keys()))
    y_out = np.array([v[0] for v in best.values()])

    # enforce sorted order (critical for PCHIP)
    idx = np.argsort(x_out)
    return x_out[idx], y_out[idx]


def candidate_matches(x, y, f0, tol=0.03):
    """
    x: detected peaks (normalized)
    y: reference lines (normalized)
    f0: coarse mapping
    tol: normalized tolerance
    """
    pairs = []
    for xi in x:
        yi_pred = f0(xi)
        idx = np.where(np.abs(y - yi_pred) < tol)[0]
        for j in idx:
            pairs.append((xi, y[j]))
    return np.array(pairs)


def universal_dispersion_calibration(
    peaks_measured,
    reference_lines,
    tol=0.03
):
    # --- Normalize ---
    x_n, x0, x1 = normalize(peaks_measured)
    y_n, y0, y1 = normalize(reference_lines)

    # --- Quantile-based initial map ---
    f0 = quantile_map(x_n, y_n)

    # --- Candidate matching (normalized space) ---
    pairs_n = candidate_matches(x_n, y_n, f0, tol=tol)
    if len(pairs_n) < 2:
        raise RuntimeError("Not enough matched peaks")

    x_fit_n = pairs_n[:, 0]
    y_fit_n = pairs_n[:, 1]

    #print(pairs_n)
    # 🔥 FIX: collapse BEFORE PCHIP
    x_fit, y_fit = collapse_candidates_pre_fit(x_fit_n, y_fit_n, f0)
    #print(x_fit, y_fit)
    # --- Robust monotone fit ---
    spline_n, inlier_mask = robust_pchip_fit(x_fit, y_fit)

    # --- Forward / inverse mappings (original units) ---
    def forward(x):
        x_nq = normalize(x)[0]
        return denormalize(spline_n(x_nq), y0, y1)

    inv_n = invert_monotone(x_fit[inlier_mask], y_fit[inlier_mask])

    def inverse(y):
        y_nq = normalize(y)[0]
        return denormalize(inv_n(y_nq), x0, x1)

    # --- Matched pairs for inspection ---

    # All candidate pairs (original units)
    pairs_all = np.column_stack([
        denormalize(x_fit, x0, x1),
        denormalize(y_fit, y0, y1)
    ])

    # Inlier pairs only (original units)
    pairs_inliers = pairs_all[inlier_mask]

    # Optional: also return normalized versions for debugging
    pairs_all_n = pairs_n

    return {
        "forward": forward,
        "inverse": inverse,
        #"pairs_all": pairs_all,
        "pairs_inliers": pairs_inliers,
        #"pairs_all_normalized": pairs_all_n,
        "pairs_inliers_normalized": [x_fit[inlier_mask], y_fit[inlier_mask]],
    }


# measured peak positions (pixels)
#pixels = np.array([120, 345, 512, 689, 842, 1030, 1210, 1390])

# known Ne lines (cm⁻1 or nm — doesn't matter)
#ne_lines = np.array([540.1, 585.2, 614.3, 640.2, 703.2, 724.5, 743.9, 748.8])

#fwd, inv = universal_dispersion_calibration(pixels, ne_lines)

# convert full spectrum
#pixel_axis = np.arange(1600)
#raman_axis = fwd(pixel_axis)

ne_peaks_cwa = [
    [533.07775,540.05616,556.27662,565.66588,571.92248,574.82985,576.44188,580.44496,582.01558,585.24878,587.28275,588.1895,590.24623,594.4834,596.5471,598.79074,602.99968,607.43376,609.6163,612.84498,614.30627,616.35937,618.2146,621.72812,626.64952,630.47893,633.44276,638.29914,640.2248,650.65277,653.28824,659.89528,667.82766,671.7043,692.94672,702.405,703.24128,705.91079,717.3938,724.51665,748.88712,753.57739,754.40439,794.31805,808.24576,811.85495,813.64061,830.03248,836.57464,837.7607,846.33569,849.53591,854.46952,857.13535,859.12583,863.46472,870.41122,877.16575,878.37539,885.38669,891.95007,898.85564,914.8672,920.17588,927.55191,930.08532,932.65072,937.33079,942.53797,945.9211,948.66825,953.4164,954.74052,966.542],
    [0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00005,0.00004,0.00005,0.00004,0.00005,0.00004,0.00004,0.00005,0.00005,0.00005,0.00004,0.00005,0.00005,0.00004,0.00005,0.00005,0.00005,0.00005,0.00005,0.0001,0.00005,0.00005,0.00005,0.00005,0.00005,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.0001,0.00004,0.00004,0.00004,0.00004,0.00004,0.00004,0.0001,0.0001,0.00004,0.00004,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.00005]
]