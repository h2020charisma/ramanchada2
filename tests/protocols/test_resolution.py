"""CWA 18133:2024 Figure 2, sections 3 & 4 -- resolution curves.

Exercises ``ramanchada2.protocols.calibration.resolution`` on the bundled ICV
BWtek 785 nm reference set (the same spectra ``test_calibrationmodel`` uses):
a real neon+silicon calibration is built and the spectral-distribution,
pixel-resolution and calcite/E2529 spectral-resolution curves are computed from
the real neon and calcite. Pure helpers and the plausibility guard are covered
deterministically alongside.
"""

import numpy as np
import pytest
import ramanchada2.misc.constants as rc2const
from ramanchada2.misc.utils.ramanshift_to_wavelength import (
    abs_nm_to_shift_cm_1,
    shift_cm_1_to_abs_nm,
)
from ramanchada2.protocols.calibration.calibration_model import CalibrationModel
from ramanchada2.protocols.calibration.resolution import (
    detect_uniform_grid,
    fit_pixel_resolution_curve,
    MIN_NEON_PEAKS,
    neon_reference_cm1,
    resolution_from_calibration,
    spectral_resolution_e2529,
)
from ramanchada2.spectrum import from_test_spe, Spectrum

LASER = 785
_FWHM_TO_SIGMA = 1.0 / 2.354820045


# --- pure helpers -----------------------------------------------------------


def test_e2529_formula():
    # ASTM E2529 (VAMAS P6): SRes = (FWHM - 0.684) / 1.0209.
    # Anchored to the published value: calcite FWHM 8.011 -> 7.177 cm-1.
    assert spectral_resolution_e2529(8.011) == pytest.approx(7.177, abs=1e-3)
    assert spectral_resolution_e2529(1.1) == pytest.approx((1.1 - 0.684) / 1.0209)


def test_detect_uniform_grid():
    assert detect_uniform_grid(np.linspace(0, 1000, 500))[0] is True
    # an axis uniform in wavelength is curved (non-uniform) in cm-1
    nm = np.linspace(
        shift_cm_1_to_abs_nm(-150, LASER), shift_cm_1_to_abs_nm(3300, LASER), 3000
    )
    assert detect_uniform_grid(np.sort(abs_nm_to_shift_cm_1(nm, LASER)))[0] is False


def test_pixel_resolution_curve_needs_six_points():
    xs = np.linspace(100, 500, 5)  # only 5 < MIN_NEON_PEAKS
    assert fit_pixel_resolution_curve(xs, np.ones_like(xs))[0] is None
    xs8 = np.linspace(100, 2000, 8)
    rng = np.random.default_rng(0)
    poly, lo, hi = fit_pixel_resolution_curve(
        xs8, 3 + 1e-4 * xs8 + rng.normal(0, 0.01, 8)
    )
    assert poly is not None
    assert lo == pytest.approx(100) and hi == pytest.approx(2000)


# --- real calibration + real calcite ----------------------------------------


def _load(sample, op="100"):
    return from_test_spe(
        sample=[sample],
        provider=["ICV"],
        device=["BWtek"],
        OP=[op],
        laser_wl=[str(LASER)],
    )


@pytest.fixture(scope="module")
def calibration():
    spe_neon = _load("Neon")
    spe_sil = _load("S0B")
    spe_neon = spe_neon.trim_axes(method="x-axis", boundaries=(100, max(spe_neon.x)))
    spe_neon = spe_neon.subtract_baseline_rc1_snip(niter=40)
    spe_sil = spe_sil.trim_axes(method="x-axis", boundaries=(520.45 - 50, 520.45 + 50))
    spe_sil = spe_sil.subtract_baseline_rc1_snip(niter=40)
    calmodel = CalibrationModel.calibration_model_factory(
        LASER,
        spe_neon,
        spe_sil,
        neon_wl=rc2const.NEON_WL[LASER],
        find_kw={"wlen": 200, "width": 1},
        should_fit=True,
        match_method="cluster",
        si_profile="Pearson4",
        interpolator_method="pchip",
    )
    assert calmodel is not None and len(calmodel.components) == 2
    return calmodel, spe_neon


def test_resolution_curves_from_real_neon_and_calcite(calibration):
    calmodel, spe_neon = calibration
    spe_calcite = _load("nCAL")
    res = resolution_from_calibration(
        calmodel,
        spe_neon,
        neon_units="cm-1",
        spe_calcite=spe_calcite,
        calcite_units="cm-1",
    )

    # Section 3: spectral distribution + pixel-resolution curve
    assert res.n_neon_peaks >= MIN_NEON_PEAKS
    assert res.curve_ok
    assert np.isfinite(res.pixel_res).any()
    assert len(res.sped) == len(res.raman_shift)

    # Section 4: the calcite ~1085.91 band was found and gave an E2529 SRes.
    # The fitted centre may sit a few cm-1 off 1085.91 -- that residual is the
    # calibration error verification exists to surface -- but the module only
    # accepts a peak within its 20 cm-1 window, so a non-None centre means the
    # right band was located.
    assert res.calcite_center is not None
    assert res.calcite_fwhm is not None and res.calcite_fwhm > 0
    assert res.spectral_resolution is not None and res.spectral_resolution > 0
    # the guard reached a verdict (either applied or withheld), never left None
    assert res.sres_plausible in (True, False)
    assert res.figure() is not None


def test_resolution_without_calcite_skips_section_four(calibration):
    calmodel, spe_neon = calibration
    res = resolution_from_calibration(calmodel, spe_neon, neon_units="cm-1")
    assert res.n_neon_peaks >= MIN_NEON_PEAKS
    assert res.spectral_resolution is None
    assert not np.isfinite(res.spectral_res).any()


# --- the plausibility guard, forced deterministically -----------------------


def _gauss(x, centers, fwhm, height=100.0, seed=0):
    sigma = fwhm * _FWHM_TO_SIGMA
    y = np.zeros_like(x, dtype=float)
    for c in centers:
        y += height * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    y += np.random.default_rng(seed).normal(0, 0.4, size=x.size)
    return Spectrum(x=np.asarray(x, dtype=float), y=y)


def _curved(lo, hi, n):
    nm = np.linspace(
        shift_cm_1_to_abs_nm(lo, LASER), shift_cm_1_to_abs_nm(hi, LASER), n
    )
    return np.sort(abs_nm_to_shift_cm_1(nm, LASER))


def test_calcite_narrower_than_neon_is_flagged_not_applied():
    """A calcite FWHM below the neon-derived instrument function is impossible
    (neon lines have ~zero intrinsic width), so the laser-effect rescale is
    withheld and no spectral-resolution curve is drawn. Driven through an
    identity model so the inputs are exactly controlled."""
    calmodel = CalibrationModel(LASER)  # no components -> apply is identity
    x_ne = _curved(-100, 3200, 3400)
    ref = neon_reference_cm1(LASER)
    ref = ref[(ref > x_ne.min() + 20) & (ref < x_ne.max() - 20)]
    neon = _gauss(x_ne, ref, fwhm=8.0)
    calcite = _gauss(_curved(950, 1250, 1200), [1085.91], fwhm=2.0, seed=1)

    res = resolution_from_calibration(
        calmodel, neon, neon_units="cm-1", spe_calcite=calcite, calcite_units="cm-1"
    )
    assert res.sres_plausible is False
    assert res.laser_effect_ratio is None
    assert not np.isfinite(res.spectral_res).any()
    assert any("implausible" in n.lower() for n in res.notes)
