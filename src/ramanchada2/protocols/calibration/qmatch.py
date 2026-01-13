from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from ramanchada2.misc.utils import argmin2d
import numpy as np
import matplotlib.pyplot as plt
import ramanchada2.misc.constants as rc2const

def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    a0, a1 = arr.min(), arr.max()
    return (arr - a0) / (a1 - a0), a0, a1

def denormalize(arr_n, a0, a1):
    return arr_n * (a1 - a0) + a0


def quantile_map(x, y, q=None):
    """
    Build coarse linear map using quantiles.
    No normalization needed!
    """
    if q is None:
        q = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    qx = np.quantile(x, q)
    qy = np.quantile(y, q)
    a = np.polyfit(qx, qy, deg=1)
    
    poly = np.poly1d(a)
    print(f"  Quantile map: y = {a[0]:.6f}*x + {a[1]:.6f}")
    
    return poly


def estimate_median_limit_from_data(A, n_sigma=3.0):
    """
    Estimate median_limit from distance distribution.
    Works directly with physical units (nm).
    """
    min_dist_per_ref = np.min(A, axis=1)
    min_dist_per_peak = np.min(A, axis=0)
    all_min_dist = np.concatenate([min_dist_per_ref, min_dist_per_peak])
    
    med = np.median(all_min_dist)
    mad = np.median(np.abs(all_min_dist - med))
    
    print(f"\nAdaptive median_limit:")
    print(f"  Median distance: {med:.2f} nm")
    print(f"  MAD: {mad:.2f} nm")
    
    if med < 1e-10 or mad < 1e-10:
        return 10.0
    
    median_limit = 1.0 + n_sigma * (mad / med)
    print(f"  median_limit = {median_limit:.2f}")
    
    return median_limit


def linear_residual_filter(x, y, n_sigma=3.0):
    """
    Filter outliers using linear fit residuals.
    Works directly in physical units.
    """
    if len(x) < 2:
        return x, y, np.ones(len(x), dtype=bool)
    
    a, b = np.polyfit(x, y, deg=1)
    y_pred = a * x + b
    resid = y - y_pred
    
    med_resid = np.median(resid)
    mad = np.median(np.abs(resid - med_resid))
    
    if mad < 1e-10:
        threshold = np.percentile(np.abs(resid), 95)
        print(f"  Linear filter: MAD too small, using 95th percentile threshold={threshold:.2f} nm")
    else:
        threshold = n_sigma * mad
        print(f"  Linear filter: MAD={mad:.2f} nm, threshold={threshold:.2f} nm")
    
    mask = np.abs(resid - med_resid) < threshold
    print(f"  Kept {mask.sum()}/{len(x)} inliers")
    
    return x[mask], y[mask], mask


def estimate_median_limit_from_data(A, n_sigma=3.0):
    """
    Estimate median_limit from distance distribution.
    Works directly with physical units (nm).
    """
    min_dist_per_ref = np.min(A, axis=1)
    min_dist_per_peak = np.min(A, axis=0)
    all_min_dist = np.concatenate([min_dist_per_ref, min_dist_per_peak])
    
    med = np.median(all_min_dist)
    mad = np.median(np.abs(all_min_dist - med))
    
    print(f"\nAdaptive median_limit:")
    print(f"  Median distance: {med:.2f} nm")
    print(f"  MAD: {mad:.2f} nm")
    
    if med < 1e-10 or mad < 1e-10:
        return 10.0
    
    median_limit = 1.0 + n_sigma * (mad / med)
    print(f"  median_limit = {median_limit:.2f}")
    
    return median_limit


def linear_residual_filter(x, y, n_sigma=3.0):
    """
    Filter outliers using linear fit residuals.
    Works directly in physical units.
    """
    if len(x) < 2:
        return x, y, np.ones(len(x), dtype=bool)
    
    a, b = np.polyfit(x, y, deg=1)
    y_pred = a * x + b
    resid = y - y_pred
    
    med_resid = np.median(resid)
    mad = np.median(np.abs(resid - med_resid))
    
    if mad < 1e-10:
        threshold = np.percentile(np.abs(resid), 95)
        print(f"  Linear filter: MAD too small, using 95th percentile threshold={threshold:.2f} nm")
    else:
        threshold = n_sigma * mad
        print(f"  Linear filter: MAD={mad:.2f} nm, threshold={threshold:.2f} nm")
    
    mask = np.abs(resid - med_resid) < threshold
    print(f"  Kept {mask.sum()}/{len(x)} inliers")
    
    return x[mask], y[mask], mask


def iterative_linear_filter(x, y, n_sigma=3.0, max_iter=5):
    """
    Iteratively fit linear model and remove outliers.
    
    Args:
        x, y: Matched pairs (normalized)
        n_sigma: Number of MAD sigmas
        max_iter: Maximum iterations
    
    Returns:
        x_inliers, y_inliers, mask
    """
    mask = np.ones(len(x), dtype=bool)
    
    for iteration in range(max_iter):
        x_fit = x[mask]
        y_fit = y[mask]
        
        if len(x_fit) < 2:
            print(f"  Iteration {iteration}: Too few points, stopping")
            break
        
        # Fit linear model on current inliers
        a, b = np.polyfit(x_fit, y_fit, deg=1)
        
        # Compute residuals on ALL points
        y_pred = a * x + b
        resid = y - y_pred
        
        # MAD on current inliers
        resid_inliers = resid[mask]
        med = np.median(resid_inliers)
        mad = np.median(np.abs(resid_inliers - med))
        
        if mad < 1e-10:
            print(f"  Iteration {iteration}: Converged (MAD={mad:.2e})")
            break
        
        threshold = n_sigma * mad
        new_mask = np.abs(resid - med) < threshold
        
        n_removed = (mask & ~new_mask).sum()
        print(f"  Iteration {iteration}: {new_mask.sum()}/{len(x)} inliers, "
              f"removed {n_removed}, MAD={mad:.6f}")
        
        if np.array_equal(mask, new_mask):
            print(f"  Converged (mask unchanged)")
            break
        
        mask = new_mask
    
    return x[mask], y[mask], mask


def universal_dispersion_calibration(
    peaks_measured,
    reference_lines,
    median_limit=None,
    n_sigma_match=3.0,
    outlier_method='linear',
    n_sigma_outlier=3.0,
    use_quantile_map=True
):
    """
    Universal dispersion calibration WITHOUT normalization.
    """
    
    # --- STEP 1: MATCHING ---
    print("\n=== STEP 1: MATCHING ===")
    
    if use_quantile_map:
        print("Using quantile map for coarse alignment...")
        f0 = quantile_map(peaks_measured, reference_lines)  # No normalization!
        A = np.abs(reference_lines[:, None] - f0(peaks_measured)[None, :])
    else:
        print("Direct matching (assuming same units)...")
        A = np.abs(reference_lines[:, None] - peaks_measured[None, :])
    
    # Distance statistics
    min_distances = np.min(A, axis=1)
    print(f"Distance range: [{min_distances.min():.2f}, {min_distances.max():.2f}] nm")
    
    if median_limit is None:
        median_limit = estimate_median_limit_from_data(A, n_sigma=n_sigma_match)
    
    print(f"Distance matrix: {A.shape} ({len(reference_lines)} ref × {len(peaks_measured)} detected)")
    print(f"Tolerance: {median_limit:.2f} × median")

    # Mutual NN matching
    matches = argmin2d(A, median_limit=median_limit)
    print(f"Mutual NN matches: {len(matches)}")
    
    if len(matches) < 2:
        raise RuntimeError("Not enough mutual NN matches")
    
    y_idx = matches[:, 0]
    x_idx = matches[:, 1]
    
    x_matched = peaks_measured[x_idx]
    y_matched = reference_lines[y_idx]
    
    # Sort by x
    idx = np.argsort(x_matched)
    x_matched = x_matched[idx]
    y_matched = y_matched[idx]
    
    print(f"\nAfter matching: {len(x_matched)} pairs")
    
    assert len(x_matched) == len(np.unique(x_matched)), "Duplicate x values!"
    assert len(y_matched) == len(np.unique(y_matched)), "Duplicate y values!"

    # --- STEP 2: OUTLIER REMOVAL ---
    print("\n=== STEP 2: OUTLIER REMOVAL ===")
    
    if outlier_method == 'linear':
        x_inliers, y_inliers, inlier_mask = linear_residual_filter(
            x_matched, y_matched, n_sigma=n_sigma_outlier
        )
    elif outlier_method == 'iterative':
        x_inliers, y_inliers, inlier_mask = iterative_linear_filter(
            x_matched, y_matched, n_sigma=n_sigma_outlier, max_iter=5
        )
    elif outlier_method is None:
        print("  Skipping outlier removal")
        x_inliers, y_inliers = x_matched, y_matched
        inlier_mask = np.ones(len(x_matched), dtype=bool)
    else:
        raise ValueError(f"Unknown outlier_method: {outlier_method}")
    
    print(f"\nAfter outlier removal: {len(x_inliers)} inliers")
    
    assert np.all(np.diff(x_inliers) > 0), "Non-monotonic x!"
    assert np.all(np.diff(y_inliers) > 0), "Non-monotonic y!"

    # --- STEP 3: INTERPOLATION ---
    print("\n=== STEP 3: INTERPOLATION ===")
    print(f"Fitting PCHIP spline on {len(x_inliers)} clean points...")
    
    spline = PchipInterpolator(x_inliers, y_inliers, extrapolate=False)

    # --- Forward mapping (no denormalization needed!) ---
    def forward(x):
        return spline(np.asarray(x, dtype=float))

    # --- Pairs for inspection ---
    pairs_all = np.column_stack([x_matched, y_matched])
    pairs_inliers = np.column_stack([x_inliers, y_inliers])

    return {
        "forward": forward,
        "pairs_all": pairs_all,
        "pairs_inliers": pairs_inliers,
    }


def diagnose_matching(peaks_measured, reference_lines, target_ref = 540, laser_wl_nm=None, use_quantile_map=True):
    """
    Diagnose why certain peaks are or aren't matched.
    
    Args:
        peaks_measured: Detected peaks (pixels or nm)
        reference_lines: Reference wavelengths (nm)
        laser_wl_nm: Optional, for context
        use_quantile_map: If True, use quantile map. If False, direct matching.
    """
    import matplotlib.pyplot as plt
    
    # Normalize
    x_n, x0, x1 = normalize(peaks_measured)
    y_n, y0, y1 = normalize(reference_lines)
    
    print("\n" + "="*70)
    print("MATCHING DIAGNOSTICS")
    print("="*70)
    
    print(f"\nDetected peaks (original units):")
    print(f"  Range: [{peaks_measured.min():.2f}, {peaks_measured.max():.2f}]")
    print(f"  Count: {len(peaks_measured)}")
    print(f"  First 5: {peaks_measured[:5]}")
    
    print(f"\nReference lines (nm):")
    print(f"  Range: [{reference_lines.min():.2f}, {reference_lines.max():.2f}]")
    print(f"  Count: {len(reference_lines)}")
    print(f"  First 5: {reference_lines[:5]}")
    
    if laser_wl_nm:
        print(f"\nLaser wavelength: {laser_wl_nm} nm")
        from ramanchada2.misc.utils.ramanshift_to_wavelength import abs_nm_to_shift_cm_1
        shifts = abs_nm_to_shift_cm_1(reference_lines[:5], laser_wl_nm)
        print(f"  First 5 ref lines as Raman shifts: {shifts}")
    
    # Build mapping and distance matrix
    if use_quantile_map:
        print(f"\nUsing quantile map for coarse alignment...")
        f0 = quantile_map(x_n, y_n)
        print(f"Quantile map: y = {f0.coefficients[0]:.6f}*x + {f0.coefficients[1]:.6f}")
        
        # Build distance matrix with quantile map
        A = np.abs(y_n[:, None] - f0(x_n)[None, :])
        
        # Show predictions for first few detected peaks
        print(f"\nPredictions for first 5 detected peaks:")
        for i in range(min(5, len(peaks_measured))):
            x_orig = peaks_measured[i]
            x_norm = x_n[i]
            y_pred_norm = f0(x_norm)
            y_pred_orig = denormalize(np.array([y_pred_norm]), y0, y1)[0]
            
            # Find nearest reference line
            distances = np.abs(reference_lines - y_pred_orig)
            nearest_idx = np.argmin(distances)
            nearest_ref = reference_lines[nearest_idx]
            distance = distances[nearest_idx]
            
            print(f"  Peak #{i}: {x_orig:.2f}")
            print(f"    → Predicted wavelength: {y_pred_orig:.2f} nm")
            print(f"    → Nearest ref line: {nearest_ref:.2f} nm (distance: {distance:.2f} nm)")
            
            if laser_wl_nm:
                shift = abs_nm_to_shift_cm_1(nearest_ref, laser_wl_nm)
                print(f"    → As Raman shift: {shift:.0f} cm⁻¹")
    else:
        print(f"\nDirect matching (no quantile map)...")
        f0 = None
        
        # Build distance matrix directly
        A = np.abs(y_n[:, None] - x_n[None, :])
        
        # Show direct distances for first few detected peaks
        print(f"\nDirect distances for first 5 detected peaks:")
        for i in range(min(5, len(peaks_measured))):
            x_orig = peaks_measured[i]
            
            # Find nearest reference line
            distances = np.abs(reference_lines - x_orig)
            nearest_idx = np.argmin(distances)
            nearest_ref = reference_lines[nearest_idx]
            distance = distances[nearest_idx]
            
            print(f"  Peak #{i}: {x_orig:.2f} nm")
            print(f"    → Nearest ref line: {nearest_ref:.2f} nm (distance: {distance:.2f} nm)")
            
            if laser_wl_nm:
                shift = abs_nm_to_shift_cm_1(nearest_ref, laser_wl_nm)
                print(f"    → As Raman shift: {shift:.0f} cm⁻¹")
    
    print(f"\nDistance matrix statistics:")
    print(f"  Shape: {A.shape} ({len(reference_lines)} refs × {len(peaks_measured)} peaks)")
    print(f"  Min distance overall: {A.min():.6f} (normalized)")
    
    # For each detected peak, show closest reference
    print(f"\nFor each detected peak, closest reference line:")
    for i in range(min(5, len(peaks_measured))):
        x_orig = peaks_measured[i]
        min_dist_norm = A[:, i].min()
        min_dist_orig = min_dist_norm * (y1 - y0)  # Convert to nm
        ref_idx = A[:, i].argmin()
        ref_line = reference_lines[ref_idx]
        
        print(f"  Peak {x_orig:.2f} → {ref_line:.2f} nm (distance: {min_dist_orig:.2f} nm)")
    
    # Check if 540 is in reference lines
    
    if target_ref in reference_lines or any(np.abs(reference_lines - target_ref) < 0.5):
        idx_target = np.argmin(np.abs(reference_lines - target_ref))
        ref_target = reference_lines[idx_target]
        print(f"\n✓ {target_ref} nm line found in references: {ref_target:.2f} nm")
        
        # Which detected peak is closest?
        distances_to_target = A[idx_target, :]
        closest_peak_idx = distances_to_target.argmin()
        closest_peak = peaks_measured[closest_peak_idx]
        distance = distances_to_target[closest_peak_idx] * (y1 - y0)
        
        print(f"  Closest detected peak: {closest_peak:.2f}")
        print(f"  Distance: {distance:.2f} nm")
        
        if use_quantile_map and f0 is not None:
            # What does quantile map predict for this peak?
            x_closest_norm = x_n[closest_peak_idx]
            y_pred_norm = f0(x_closest_norm)
            y_pred_orig = denormalize(np.array([y_pred_norm]), y0, y1)[0]
            
            print(f"  Quantile map predicts: {y_pred_orig:.2f} nm for this peak")
            print(f"  Error: {abs(y_pred_orig - ref_target):.2f} nm")
    else:
        print(f"\n✗ {target_ref} nm line NOT in reference lines!")
        print(f"  Reference range: {reference_lines.min():.1f} - {reference_lines.max():.1f} nm")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mapping (quantile or direct)
    if use_quantile_map and f0 is not None:
        predicted_wl = denormalize(f0(x_n), y0, y1)
        ax1.scatter(peaks_measured, predicted_wl, 
                    label='Detected peaks → Predicted wavelength', 
                    alpha=0.6, s=50)
        ax1.set_title('Quantile Map Predictions')
    else:
        ax1.scatter(peaks_measured, peaks_measured, 
                    label='Detected peaks (direct matching)', 
                    alpha=0.6, s=50)
        ax1.set_title('Direct Matching (1:1 line)')
    
    ax1.hlines(reference_lines, peaks_measured.min(), peaks_measured.max(), 
               colors='red', alpha=0.3, linewidth=0.5, label='Reference lines')
    ax1.set_xlabel('Detected Peak Position')
    ax1.set_ylabel('Wavelength (nm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight 540 if present
    if target_ref in reference_lines or any(np.abs(reference_lines - target_ref) < 0.5):
        ax1.axhline(ref_target, color='green', linewidth=2, label=f'{ref_target:.1f} nm', linestyle='--')
    
    # Plot 2: Distance matrix heatmap
    n_show = min(20, len(peaks_measured))
    m_show = min(30, len(reference_lines))
    
    im = ax2.imshow(A[:m_show, :n_show], aspect='auto', cmap='viridis', 
                    interpolation='nearest')
    ax2.set_xlabel('Detected Peak Index')
    ax2.set_ylabel('Reference Line Index')
    
    if use_quantile_map:
        ax2.set_title(f'Distance Matrix with Quantile Map\n(first {n_show} peaks, {m_show} refs)')
    else:
        ax2.set_title(f'Distance Matrix (Direct)\n(first {n_show} peaks, {m_show} refs)')
    
    plt.colorbar(im, ax=ax2, label='Distance (normalized)')
    
    # Mark 540 nm if present
    if target_ref in reference_lines or any(np.abs(reference_lines - target_ref) < 0.5):
        if idx_target < m_show:
            ax2.axhline(idx_target, color='red', linewidth=2, alpha=0.7)
            ax2.text(n_show-1, idx_target, f'  {ref_target:.1f}nm', 
                    color='red', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('matching_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved visualization to: matching_diagnosis.png")
    
    return fig