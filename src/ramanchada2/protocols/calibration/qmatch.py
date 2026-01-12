from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from ramanchada2.misc.utils import argmin2d
import numpy as np
import matplotlib.pyplot as plt


def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    a0, a1 = arr.min(), arr.max()
    return (arr - a0) / (a1 - a0), a0, a1

def denormalize(arr_n, a0, a1):
    return arr_n * (a1 - a0) + a0


def quantile_map(x, y, q=(0.05,  0.5,  0.95)):
    """
    Build coarse linear map using quantiles.
    """
    qx = np.quantile(x, q)
    qy = np.quantile(y, q)
    a = np.polyfit(qx, qy, deg=1)
    return np.poly1d(a)


def robust_pchip_fit(x, y, max_iter=5, sigma=0.02):
    """
    Iterative robust monotone fit.
    """
    mask = np.ones(len(x), dtype=bool)

    for _ in range(max_iter):
        spline = PchipInterpolator(x[mask], y[mask], extrapolate=False)
        resid = y - spline(x)
        mad = np.median(np.abs(resid))
        mask = np.abs(resid) < sigma * max(mad, 1e-6)
        print(mad, len(mask))
    return x[mask], y[mask], mask


def robust_pchip_fit_simple(x, y, n_sigma=3.0):
    """
    Single-pass robust fit.
    """
    # Initial fit on all points
    spline = PchipInterpolator(x, y, extrapolate=False)
    resid = y - spline(x)
    
    # Detect outliers
    mad = np.median(np.abs(resid))
    threshold = n_sigma * mad
    mask = np.abs(resid) <= threshold
    
    print(f"  Outlier removal: {mask.sum()}/{len(x)} inliers, "
          f"MAD={mad:.6f}, threshold={threshold:.6f}")
    
    # Refit without outliers
    return PchipInterpolator(x[mask], y[mask], extrapolate=False), mask


def robust_pchip_fit_percentile(x, y, max_iter=3, percentile=90):
    """
    Iterative robust fit keeping best X% of points by residual.
    """
    mask = np.ones(len(x), dtype=bool)
    
    for iteration in range(max_iter):
        x_fit = x[mask]
        y_fit = y[mask]
        
        if len(x_fit) < 3:
            break
        
        spline = PchipInterpolator(x_fit, y_fit, extrapolate=True)
        resid = np.abs(y - spline(x))
        
        # Keep best percentile
        threshold = np.percentile(resid, percentile)
        new_mask = resid <= threshold
        
        n_inliers = new_mask.sum()
        print(f"  Iteration {iteration}: {n_inliers}/{len(x)} inliers, "
              f"{percentile}th percentile threshold={threshold:.6f}")
        
        if np.array_equal(mask, new_mask):
            print(f"  Converged")
            break
        
        mask = new_mask
    
    return x[mask], y[mask], mask

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


def estimate_median_limit_from_data(A, n_sigma=3.0):
    """
    Estimate median_limit: multiplier such that threshold = median × median_limit
    """
    # Minimum distances
    min_dist_per_ref = np.min(A, axis=1)
    min_dist_per_peak = np.min(A, axis=0)
    all_min_dist = np.concatenate([min_dist_per_ref, min_dist_per_peak])
    
    med = np.median(all_min_dist)
    mad = np.median(np.abs(all_min_dist - med))
    
    print(f"\nAdaptive median_limit:")
    print(f"  Median distance: {med:.6f}")
    print(f"  MAD: {mad:.6f}")
    
    if med < 1e-10 or mad < 1e-10:
        print(f"  Using default: 10.0")
        return 10.0
    
    # median_limit = (med + n_sigma*mad) / med = 1 + n_sigma*(mad/med)
    median_limit = 1.0 + n_sigma * (mad / med)
    
    print(f"  median_limit = 1 + {n_sigma} × (MAD/median) = {median_limit:.2f}")
    print(f"Keep distances ≤ {median_limit:.2f} × median")
    
    return median_limit


def universal_dispersion_calibration(
    peaks_measured,
    reference_lines,
    median_limit=10.0
):
    # --- Normalize ---
    x_n, x0, x1 = normalize(peaks_measured)
    y_n, y0, y1 = normalize(reference_lines)

    # --- Quantile-based initial map ---
    f0 = quantile_map(x_n, y_n)

    # --- Use mutual nearest neighbors instead of tolerance matching ---
    
    # Build distance matrix: A[i,j] = error for matching y[i] to x[j]
    # Error = |y[i] - f0(x[j])|
    A = np.abs(y_n[:, None] - f0(x_n)[None, :])
    min_distances = np.min(A, axis=1)
    plt.hist(min_distances)
    print(f"{min(min_distances) , max(min_distances)}")
    if median_limit is None:
        median_limit = estimate_median_limit_from_data(A, n_sigma=6)
    
    print(f"Distance matrix shape: {A.shape} ({len(y_n)} ref lines × {len(x_n)} detected peaks) tollerance {median_limit}")

    # Find mutual nearest neighbors
    matches = argmin2d(A, median_limit=median_limit)
    
    print(f"Mutual NN matches: {len(matches)}")
    
    if len(matches) < 2:
        raise RuntimeError("Not enough mutual NN matches")
    
    # Extract matched pairs (y_idx, x_idx)
    y_idx = matches[:, 0]
    x_idx = matches[:, 1]
    
    x_fit_n = x_n[x_idx]
    y_fit_n = y_n[y_idx]

    pairs_n = np.column_stack([x_fit_n, y_fit_n])

    
    # Sort by x
    idx = np.argsort(x_fit_n)
    x_fit = x_fit_n[idx]
    y_fit = y_fit_n[idx]
    
    print(f"Final pairs: {len(x_fit)}")
    print(f"  Unique x: {len(np.unique(x_fit))}")
    print(f"  Unique y: {len(np.unique(y_fit))}")
    
    # Verify no duplicates
    assert len(x_fit) == len(np.unique(x_fit)), "Duplicate x values!"
    assert len(y_fit) == len(np.unique(y_fit)), "Duplicate y values!"

    # --- Robust monotone fit ---
    x, y, inlier_mask =  robust_pchip_fit_percentile(x_fit, y_fit)
    spline_n = PchipInterpolator(x, y, extrapolate=True)

    # --- Forward / inverse mappings (original units) ---
    def forward(x):
        x = np.asarray(x, dtype=float)
        x_nq = (x - x0) / (x1 - x0)     # use calibration normalization
        return denormalize(spline_n(x_nq), y0, y1)

    #def inverse(y):
    #    y = np.asarray(y, dtype=float)
    #    y_nq = (y - y0) / (y1 - y0)
    #    return denormalize(inv_n(y_nq), x0, x1)

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
        #"inverse": inverse,
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
