"""Tests for qmatch.robust_poly_residual_filter, in particular the opt-in
``auto_reduce_degree`` behaviour for sparse reference sets.

Context: CAL (calcite) verification uses only 6 certified reference positions
(155.21, 281.26, 711.95, 1085.91, 1435.22, 1748.91 cm-1), unlike neon's ~15-30
NIST lines. The default degree-2 RANSAC filter draws 3-point samples to fit a
quadratic; with only 6 points and 2 of them mismatched to the wrong band, a
3-point sample can include a bad point and still look self-consistent, so the
"largest consensus" is not a reliable signal and the filter can fail to reject
a mismatch that is tens of cm-1 off (observed on P6_0701/OP4's CAL peaks:
matched_peaks_samples.csv carries an inlier_mask=True row at 181 cm-1 from
this exact reference set).

``auto_reduce_degree=True`` recurses to a lower polynomial degree when there
are too few points for the requested degree to be well-determined, down to a
plain MAD-distance test at degree 0. It defaults to False so the neon
calibration path (which has enough points for degree-2 RANSAC to be reliable)
is unaffected.
"""
import numpy as np
import pytest

from ramanchada2.protocols.calibration.qmatch import robust_poly_residual_filter

# The real CAL reference table from calibration_verify.py's get_reference_peaks("CAL").
CAL_REFERENCE_CM1 = np.array(
    [155.21, 281.26, 711.95, 1085.91, 1435.22, 1748.91])


def _cal_like_mismatch():
    """Reproduce the shape of the P6_0701/OP4 CAL failure: two reference
    positions matched to the wrong spectral peak, tens of cm-1 away, the rest
    well matched."""
    y = CAL_REFERENCE_CM1.copy()
    x = y.copy()
    x[1] = 462.63    # 281.26 matched to a peak that is really ~181 cm-1 away
    x[4] = 1506.66    # 1435.22 matched to a peak that is really ~71 cm-1 away
    return x, y


def test_default_behaviour_is_unchanged_for_dense_reference_sets():
    """auto_reduce_degree defaults to False: a neon-sized reference set (well
    above the degree-2 threshold) must produce identical output whether or not
    the new parameter is mentioned, so the calibration path is untouched."""
    rng = np.random.default_rng(1)
    y = np.linspace(540, 700, 20)
    x = y + rng.normal(0, 0.05, size=y.size)

    x1, y1, mask1 = robust_poly_residual_filter(x, y, deg=2, n_sigma=3)
    x2, y2, mask2 = robust_poly_residual_filter(
        x, y, deg=2, n_sigma=3, auto_reduce_degree=False)

    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_sparse_set_without_auto_reduce_can_miss_mismatches():
    """Pin the failure mode this fix addresses: with auto_reduce_degree left
    at its default False, the degree-2 filter on a 6-point CAL-like set is not
    guaranteed to reject a badly mismatched pair. This test documents the
    defect rather than requiring it (the RANSAC search is a bit stochastic),
    so it only asserts the filter runs and returns a well-formed mask."""
    x, y = _cal_like_mismatch()
    xi, yi, mask = robust_poly_residual_filter(x, y, deg=2, n_sigma=3)
    assert mask.shape == (6,)
    assert mask.sum() >= 1


def test_auto_reduce_degree_rejects_gross_mismatch_on_sparse_set():
    """With auto_reduce_degree=True, at least the grossly mismatched point
    (index 1: 281.26 -> matched 181 cm-1 away, the largest deviation in the
    set) must be flagged as an outlier on the real CAL reference geometry."""
    x, y = _cal_like_mismatch()
    xi, yi, mask = robust_poly_residual_filter(
        x, y, deg=2, n_sigma=3, auto_reduce_degree=True)
    assert mask.shape == (6,)
    assert not mask[1], "the ~181 cm-1 mismatch must be rejected"


def test_auto_reduce_degree_keeps_well_matched_sparse_set_intact():
    """A sparse but well-matched reference set must not lose points just
    because auto_reduce_degree is enabled - the fallback should not become
    more aggressive than necessary when there is nothing to reject."""
    rng = np.random.default_rng(2)
    y = CAL_REFERENCE_CM1.copy()
    x = y + rng.normal(0, 0.05, size=y.size)

    xi, yi, mask = robust_poly_residual_filter(
        x, y, deg=2, n_sigma=3, auto_reduce_degree=True)
    assert mask.all(), "no genuine mismatch should not be manufactured"


def test_auto_reduce_degree_zero_uses_mad_distance_base_case():
    """When even a degree-1 fit is underdetermined, the degree-0 base case (a
    plain MAD-distance test around the median) must still flag an obvious
    outlier and keep the well-behaved points."""
    y = np.array([100.0, 102.0, 98.0, 250.0])  # last point is the outlier
    x = y.copy()
    xi, yi, mask = robust_poly_residual_filter(
        x, y, deg=1, n_sigma=3, auto_reduce_degree=True)
    assert mask.shape == (4,)
    assert not mask[3]
    assert mask[:3].all()


def test_small_n_still_short_circuits_before_any_fit():
    """n <= deg + 2 keeps returning all points as inliers regardless of
    auto_reduce_degree - there still is not enough data to say anything."""
    y = np.array([100.0, 300.0])
    x = y.copy()
    for flag in (False, True):
        xi, yi, mask = robust_poly_residual_filter(
            x, y, deg=2, n_sigma=3, auto_reduce_degree=flag)
        assert mask.all()
