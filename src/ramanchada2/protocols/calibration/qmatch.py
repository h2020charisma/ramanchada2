from ramanchada2.misc.utils import argmin2d
import numpy as np
import matplotlib.pyplot as plt
from ramanchada2.protocols.calibration.interpolators import (
    get_interpolator
)
import logging

logger = logging.getLogger(__name__)


def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    a0, a1 = arr.min(), arr.max()
    return (arr - a0) / (a1 - a0), a0, a1


def denormalize(arr_n, a0, a1):
    return arr_n * (a1 - a0) + a0


def find_closest_pairs_quantile_idx(x, y, **kw_args):
    f0 = quantile_map(x, y)  # No normalization!
    A = np.abs(y[:, None] - f0(x)[None, :])
    median_limit = estimate_median_limit_from_data(A, **kw_args)
    matches = argmin2d(A, median_limit=median_limit)
    logger.debug(f"Mutual NN matches: {len(matches)}")
    y_idx = matches[:, 0]
    x_idx = matches[:, 1]

    logger.debug(f"x {len(x)} y {len(y)}")
    return x_idx, y_idx


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
    logger.info(f"  Quantile map: y = {a[0]:.6f}*x + {a[1]:.6f}")

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

    logger.info(f"\nAdaptive median_limit: Median distance: {med:.2f} nm MAD: {mad:.2f} nm")

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


def robust_poly_residual_filter(x, y, deg=2, n_sigma=3.0, n_iter=300, seed=0,
                                auto_reduce_degree=False):
    """Curve-aware robust outlier filter for matched (measured, reference) pairs.

    ``linear_residual_filter`` fits a *straight line* and rejects points by their
    residual from it. But the measured->reference relation of a Raman spectrograph is a
    gently curved dispersion, so correctly-matched lines near the laser deviate from the
    global straight line and get discarded. Dropping them shrinks the calibration anchor
    span, which in turn pushes the Silicon laser-zeroing peak (and low-wavenumber sample
    peaks) *outside* the anchor range where they are extrapolated -> the whole cm-1 axis
    is mis-scaled and calibration worsens the spectra.

    This filter keeps the largest RANSAC consensus of points lying within an adaptive band
    of a degree-``deg`` fit, so on-trend (curved) near-laser lines survive while genuine
    mismatches are still rejected. Returns ``(x_inliers, y_inliers, mask)`` with ``mask``
    aligned to the input order, matching ``linear_residual_filter``.

    ``auto_reduce_degree`` (default ``False``, so neon-sized reference sets keep today's
    exact behaviour): when ``True`` and ``n`` is too small relative to ``deg`` for a
    degree-``deg`` RANSAC sample to be statistically meaningful (fewer than
    ``2 * (deg + 2)`` points -- e.g. a degree-2 fit through 3 of only 6 points, CAL/PST-
    sized reference sets, unlike neon's ~15-30), recurse to one degree lower, which needs
    fewer points to be well-determined. A degree-0 fit (a single reference value) cannot
    discriminate a mismatch by curve shape at all, so a pure MAD-distance rejection is used
    as the base case.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= deg + 2:
        return x, y, np.ones(n, dtype=bool)

    if auto_reduce_degree and deg > 0 and n < 2 * (deg + 2):
        return robust_poly_residual_filter(
            x, y, deg=deg - 1, n_sigma=n_sigma, n_iter=n_iter, seed=seed,
            auto_reduce_degree=auto_reduce_degree)
    if auto_reduce_degree and deg == 0 and n < 2 * (deg + 2):
        r = y - np.median(y)
        mad = np.median(np.abs(r - np.median(r)))
        thr = max(n_sigma * 1.4826 * mad, 1e-9)
        mask = np.abs(r) < thr
        if mask.sum() < 2:
            mask = np.ones(n, dtype=bool)
        print(f"  Robust deg-0 filter (n={n} too small for RANSAC): "
              f"band={thr:.3f}, kept {mask.sum()}/{n} inliers")
        return x[mask], y[mask], mask

    xc = x - x.mean()  # centre for numerical stability of polyfit
    # Initial (generous) acceptance band for the RANSAC search: n_sigma * robust-sigma of
    # residuals to a curve-aware (deg) fit, floored by a fraction of the reference spacing
    # so a near-linear region with a tiny MAD does not clip the genuinely curved tails.
    c_all = np.polyfit(xc, y, deg)
    r_all = y - np.polyval(c_all, xc)
    mad = np.median(np.abs(r_all - np.median(r_all)))
    spacing = np.median(np.abs(np.diff(np.sort(y))))
    thr = max(n_sigma * 1.4826 * mad, 0.25 * spacing)
    if thr <= 0:
        thr = np.percentile(np.abs(r_all), 95)

    rng = np.random.default_rng(seed)
    best = np.zeros(n, dtype=bool)
    for _ in range(n_iter):
        idx = rng.choice(n, deg + 1, replace=False)
        try:
            c = np.polyfit(xc[idx], y[idx], deg)
        except Exception:
            continue
        m = np.abs(y - np.polyval(c, xc)) < thr
        if m.sum() > best.sum():
            best = m
    if best.sum() >= deg + 2:  # refit on the consensus and re-select
        c = np.polyfit(xc[best], y[best], deg)
        best = np.abs(y - np.polyval(c, xc)) < thr

    # Doublet-misassignment split: when the matcher alternates between the components of
    # blended reference doublets, the consensus residuals form TWO parallel populations
    # (~a fraction of a line spacing apart). Fitting through both tilts the model; keep
    # the majority population, which the sample-peak residuals confirm as the correct
    # assignment (the minority would shift the axis by tens of cm-1).
    if best.sum() >= 2 * (deg + 2):
        c = np.polyfit(xc[best], y[best], deg)
        rb = np.sort((y - np.polyval(c, xc))[best])
        split_gap, split_at = 0.0, None
        for k in range(deg + 2, len(rb) - (deg + 2) + 1):
            a, b = rb[:k], rb[k:]
            gap = b[0] - a[-1]
            scatter = max(np.median(np.abs(a - np.median(a))),
                          np.median(np.abs(b - np.median(b))), 1e-6)
            if gap > 3.0 * 1.4826 * scatter and gap > split_gap:
                split_gap, split_at = gap, (a[-1] + b[0]) / 2.0
        if split_at is not None:
            r_all_c = y - np.polyval(c, xc)
            lower = best & (r_all_c < split_at)
            upper = best & (r_all_c >= split_at)
            major = lower if lower.sum() >= upper.sum() else upper
            if major.sum() >= deg + 2:
                print(f"  Robust deg-{deg} filter: bimodal residuals (gap={split_gap:.3f}), "
                      f"keeping majority {major.sum()}/{best.sum()}")
                best = major

    # Tighten the band to the consensus scatter. The RANSAC band is floored by the
    # reference line spacing, so near-miss mismatches (a fraction of a line spacing off,
    # e.g. spurious peaks matched to the nearest available line) survive the search and
    # tilt the subsequent model fit -- the exact "slope error amplified by Si zeroing"
    # failure. The consensus residual MAD is the real measurement noise; rescale to it,
    # with a small floor so curved tails beyond the deg-fit are not clipped.
    if best.sum() >= deg + 2:
        c = np.polyfit(xc[best], y[best], deg)
        r = y - np.polyval(c, xc)
        mad_c = np.median(np.abs(r[best] - np.median(r[best])))
        thr2 = min(thr, max(n_sigma * 1.4826 * mad_c, 0.05 * spacing))
        tight = np.abs(r) < thr2
        if tight.sum() >= deg + 2:
            c = np.polyfit(xc[tight], y[tight], deg)  # final refit on the tightened set
            tight = np.abs(y - np.polyval(c, xc)) < thr2
            if tight.sum() >= deg + 2:
                best, thr = tight, thr2

    print(f"  Robust deg-{deg} filter: band={thr:.3f}, kept {best.sum()}/{n} inliers")
    return x[best], y[best], best


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
            print("  Converged (mask unchanged)")
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
    use_quantile_map=True,
    interpolator="poly"
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
        x_inliers, y_inliers, _ = linear_residual_filter(
            x_matched, y_matched, n_sigma=n_sigma_outlier
        )
    elif outlier_method == 'iterative':
        x_inliers, y_inliers, _ = iterative_linear_filter(
            x_matched, y_matched, n_sigma=n_sigma_outlier, max_iter=5
        )
    elif outlier_method is None:
        print("  Skipping outlier removal")
        x_inliers, y_inliers = x_matched, y_matched
    else:
        raise ValueError(f"Unknown outlier_method: {outlier_method}")

    print(f"\nAfter outlier removal: {len(x_inliers)} inliers")

    assert np.all(np.diff(x_inliers) > 0), "Non-monotonic x!"
    assert np.all(np.diff(y_inliers) > 0), "Non-monotonic y!"

    # --- STEP 3: INTERPOLATION ---
    logger.info("\n=== STEP 3: INTERPOLATION ===")
    print(f"Fitting {interpolator} on {len(x_inliers)} clean points...")

    interp = get_interpolator(x_inliers, y_inliers, interpolator_method=interpolator)
    print(interp)
    # --- Pairs for inspection ---
    pairs_all = np.column_stack([x_matched, y_matched])
    pairs_inliers = np.column_stack([x_inliers, y_inliers])

    return {
        "interpolator": interp,
        "pairs_all": pairs_all,
        "pairs_inliers": pairs_inliers,
    }


def diagnose_matching(peaks_measured, reference_lines, target_ref=540,
                      laser_wl_nm=None, use_quantile_map=True,
                      output_file="matching_diagnosis.png"):
    """
    Diagnose why certain peaks are or aren't matched.

    Args:
        peaks_measured: Detected peaks (pixels or nm)
        reference_lines: Reference wavelengths (nm)
        target_ref: Target reference line to highlight (default 540 nm)
        laser_wl_nm: Optional, for context
        use_quantile_map: If True, use quantile map. If False, direct matching.
    """
    print("\n" + "="*70)
    print("MATCHING DIAGNOSTICS")
    print("="*70)

    print("\nDetected peaks:")
    print(f"  Range: [{peaks_measured.min():.2f}, {peaks_measured.max():.2f}]")
    print(f"  Count: {len(peaks_measured)}")
    print(f"  First 5: {peaks_measured[:5]}")

    print("\nReference lines (nm):")
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
        print("\nUsing quantile map for coarse alignment...")
        f0 = quantile_map(peaks_measured, reference_lines)
        print(f"Quantile map: y = {f0.coefficients[0]:.6f}*x + {f0.coefficients[1]:.6f}")

        # Build distance matrix with quantile map
        A = np.abs(reference_lines[:, None] - f0(peaks_measured)[None, :])

        # Show predictions for first few detected peaks
        print("\nPredictions for first 5 detected peaks:")
        for i in range(min(5, len(peaks_measured))):
            x_orig = peaks_measured[i]
            y_pred = f0(x_orig)

            # Find nearest reference line
            distances = np.abs(reference_lines - y_pred)
            nearest_idx = np.argmin(distances)
            nearest_ref = reference_lines[nearest_idx]
            distance = distances[nearest_idx]

            print(f"  Peak #{i}: {x_orig:.2f}")
            print(f"    → Predicted wavelength: {y_pred:.2f} nm")
            print(f"    → Nearest ref line: {nearest_ref:.2f} nm (distance: {distance:.2f} nm)")

            if laser_wl_nm:
                shift = abs_nm_to_shift_cm_1(nearest_ref, laser_wl_nm)
                print(f"    → As Raman shift: {shift:.0f} cm⁻¹")
    else:
        print("\nDirect matching (no quantile map)...")
        f0 = None

        # Build distance matrix directly
        A = np.abs(reference_lines[:, None] - peaks_measured[None, :])

        # Show direct distances for first few detected peaks
        print("\nDirect distances for first 5 detected peaks:")
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

    print("\nDistance matrix statistics:")
    print(f"  Shape: {A.shape} ({len(reference_lines)} refs × {len(peaks_measured)} peaks)")
    print(f"  Min distance overall: {A.min():.2f} nm")

    # For each detected peak, show closest reference
    print("\nFor each detected peak, closest reference line:")
    for i in range(min(5, len(peaks_measured))):
        x_orig = peaks_measured[i]
        min_dist = A[:, i].min()
        ref_idx = A[:, i].argmin()
        ref_line = reference_lines[ref_idx]

        print(f"  Peak {x_orig:.2f} → {ref_line:.2f} nm (distance: {min_dist:.2f} nm)")

    # Check if target reference is in reference lines
    if target_ref in reference_lines or any(np.abs(reference_lines - target_ref) < 0.5):
        idx_target = np.argmin(np.abs(reference_lines - target_ref))
        ref_target = reference_lines[idx_target]
        print(f"\n✓ {target_ref} nm line found in references: {ref_target:.2f} nm")

        # Which detected peak is closest?
        distances_to_target = A[idx_target, :]
        closest_peak_idx = distances_to_target.argmin()
        closest_peak = peaks_measured[closest_peak_idx]
        distance = distances_to_target[closest_peak_idx]

        print(f"  Closest detected peak: {closest_peak:.2f}")
        print(f"  Distance: {distance:.2f} nm")

        if use_quantile_map and f0 is not None:
            # What does quantile map predict for this peak?
            y_pred = f0(closest_peak)

            print(f"  Quantile map predicts: {y_pred:.2f} nm for this peak")
            print(f"  Error: {abs(y_pred - ref_target):.2f} nm")
    else:
        print(f"\n✗ {target_ref} nm line NOT in reference lines!")
        print(f"  Reference range: {reference_lines.min():.1f} - {reference_lines.max():.1f} nm")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mapping (quantile or direct)
    if use_quantile_map and f0 is not None:
        predicted_wl = f0(peaks_measured)
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

    # Highlight target reference if present
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

    plt.colorbar(im, ax=ax2, label='Distance (nm)')

    # Mark target reference if present
    if target_ref in reference_lines or any(np.abs(reference_lines - target_ref) < 0.5):
        if idx_target < m_show:
            ax2.axhline(idx_target, color='red', linewidth=2, alpha=0.7)
            ax2.text(n_show-1, idx_target, f'  {ref_target:.1f}nm',
                     color='red', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved visualization to: {output_file}")

    return fig
