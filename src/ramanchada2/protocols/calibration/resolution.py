"""CWA 18133:2024 Figure 2, sections 3 & 4 — resolution curves.

Section 3: the **spectral distribution** curve (Raman-shift width per pixel of
the calibrated x-axis) and the **pixel resolution** curve (Gaussian-fit neon
peak FWHM vs position on the calibrated Raman-shift axis).
Section 4: the **spectral resolution** from the calcite ~1085.91 cm-1 peak FWHM
(Voigt fit, ASTM E2529 formula), the spectral-resolution curve and the
SpeD:SRes curve.

This is the per-spectrum core of the CHARISMA/VAMAS ``spectraframe_resolution``
task, lifted out of its batch harness so it can be called on a single calibrated
neon spectrum (plus an optional calcite spectrum) given a fitted
:class:`~ramanchada2.protocols.calibration.calibration_model.CalibrationModel`.
It takes a live model, not a file, so it is indifferent to how the model was
persisted (pickle or JSON ``to_dict``/``from_dict``).

Only the x-calibration model is used; y-calibration is intentionally not applied
(CWA sections 3-4 are x-axis only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

import ramanchada2.misc.constants as rc2const
from ramanchada2.misc.types.peak_candidates import ListPeakCandidateMultiModel
from ramanchada2.misc.utils.ramanshift_to_wavelength import (
    abs_nm_to_shift_cm_1,
    shift_cm_1_to_abs_nm,
)
from ramanchada2.spectrum import Spectrum

if TYPE_CHECKING:  # avoid any import cycle at module load
    from ramanchada2.protocols.calibration.calibration_model import CalibrationModel

# CWA 18133 Table 7 (RR1 calcite study); ASTM E2529 uses the ~1085 cm-1 band
CALCITE_REF_CM1 = 1085.91
CALCITE_WINDOW_CM1 = 100
# CWA 18133 Table 1 boundary of use
PIXEL_RESOLUTION_BOUNDARY_NM = 0.8
# max distance between a fitted peak and a NIST neon line to accept it
NEON_MATCH_TOL_CM1 = 10
# min fitted neon peaks required to trust a resolution curve (below this the
# polynomial is under-determined and the curve is not drawn / not enveloped)
MIN_NEON_PEAKS = 6
# resolution curves are only evaluated within the neon peak span, widened by
# this fraction of the span (small margin, no far extrapolation)
CURVE_MARGIN_FRAC = 0.05
# raw x-axis spacing with relative spread below this is a vendor-resampled
# (uniform) grid: the spectral distribution then shows the export grid, not
# the physical detector pixels
UNIFORM_GRID_REL_TOL = 0.01
# the spectral resolution cannot be meaningfully better than the neon-derived
# pixel resolution (neon lines have ~zero intrinsic width, so their FWHM *is*
# the instrument function); a laser-effect ratio below this bound - allowing
# for the ~20% stated accuracy of the ASTM E2529 formula - means the calcite
# fit is defective and the rescale is not applied
SRES_MIN_RATIO = 0.8
# ASTM E2529 spectral-resolution formula as applied in the VAMAS P6 analysis
# (cross-checked against ASTM E2529): the calcite 1085 band's intrinsic width
# is subtracted (offset 0.684) and the instrument response scaled (slope
# 1.0209), so SRes = (FWHM - 0.684) / 1.0209. FWHM 8.011 -> 7.177 cm-1.
# (Not FWHM/0.684 - 1.029, which swaps the two roles.)
E2529_OFFSET = 0.684
E2529_SLOPE = 1.0209
# default peak-finding window (VAMAS-tuned; see the calibration engine)
DEFAULT_FIND_KW = {"wlen": 200, "width": 1}

_SEAM_NOTE = (
    "Narrow dips at a handful of regularly-spaced points in the spectral "
    "distribution curve mark detector segment-stitching seams already present "
    "in the raw spectrum, not a calibration or fitting defect - the "
    "pixel/spectral resolution curves are fit through the neon peaks and are "
    "not affected by them."
)
_UNIFORM_GRID_NOTE = (
    "The raw spectrum of this instrument is on a uniform grid "
    "(vendor-resampled export), so the flat spectral distribution curve shows "
    "the resampling grid, not the physical pixel pitch; the SpeD and SpeD:SRes "
    "curves must not be interpreted as CWA pixel properties."
)
_IMPLAUSIBLE_NOTE = (
    "The calcite fit is implausible (its E2529 spectral resolution falls well "
    "below the neon-derived instrument function), so the laser-effect rescale "
    "was not applied and no spectral resolution curve is drawn."
)


def detect_uniform_grid(x, rel_tol=UNIFORM_GRID_REL_TOL):
    """``(is_uniform, median_step)`` of a raw x-axis. A (near-)constant spacing
    means the vendor export was resampled onto a uniform grid."""
    d = np.diff(np.asarray(x, dtype=float))
    med = float(np.median(d))
    if len(d) < 2 or med == 0:
        return False, med
    return bool((d.max() - d.min()) / abs(med) < rel_tol), med


def fwhm_cm1_to_nm(center_cm1, fwhm_cm1, laser_wl):
    """FWHM expressed on the wavelength axis at the given peak position."""
    lo = shift_cm_1_to_abs_nm(center_cm1 - fwhm_cm1 / 2, laser_wl)
    hi = shift_cm_1_to_abs_nm(center_cm1 + fwhm_cm1 / 2, laser_wl)
    return abs(hi - lo)


def spectral_distribution(spe_calibrated):
    """CWA 18133 3.1.9: the Raman-shift width collected by pixel n, taken as
    ``halfway(n, n+1) - halfway(n-1, n) == np.gradient``."""
    x = spe_calibrated.x
    return x, np.gradient(x)


def neon_reference_cm1(laser_wl):
    """NIST neon lines (nm) converted to Raman shift for this laser."""
    lines_nm = np.array(sorted(rc2const.NEON_WL[laser_wl].keys()))
    return np.sort(abs_nm_to_shift_cm_1(lines_nm, laser_wl))


def fit_neon_peaks(spe_ne_cal, laser_wl, find_kw=None, prominence_coeff=3):
    """Pixel-resolution points: Gaussian fit of the neon peaks on the calibrated
    Raman-shift axis. Only candidate groups near a NIST neon line are fitted, and
    each reference line keeps its single best (highest) fitted peak - otherwise
    noise bumps distort the resolution curve."""
    ref_cm1 = neon_reference_cm1(laser_wl)
    find_kw = dict(find_kw or DEFAULT_FIND_KW)
    find_kw["prominence"] = spe_ne_cal.y_noise_MAD() * prominence_coeff
    cand = spe_ne_cal.find_peak_multipeak(**find_kw)
    groups = [
        g
        for g in cand
        if np.min(np.abs(np.asarray(g.positions)[:, None] - ref_cm1[None, :]))
        < NEON_MATCH_TOL_CM1
    ]
    if not groups:
        return pd.DataFrame(columns=["center", "fwhm", "fwhm_stderr", "height"])
    fitres = spe_ne_cal.fit_peak_multimodel(
        profile="Gaussian",
        candidates=ListPeakCandidateMultiModel(root=groups),
        no_fit=False,
        bound_centers_to_group=True,
        vary_baseline=False,
    )
    df_peaks = fitres.to_dataframe_peaks()
    df_peaks = df_peaks.loc[
        np.isfinite(df_peaks["fwhm"])
        & (df_peaks["fwhm"] > 0)
        & np.isfinite(df_peaks["center"])
        & (df_peaks["center"] >= min(spe_ne_cal.x))
        & (df_peaks["center"] <= max(spe_ne_cal.x))
    ]
    if "fwhm_stderr" in df_peaks.columns:
        df_peaks = df_peaks.loc[
            df_peaks["fwhm_stderr"].isna()
            | (df_peaks["fwhm_stderr"] < df_peaks["fwhm"])
        ]
    if df_peaks.empty:
        return df_peaks
    # one fitted peak per NIST line: nearest line within tolerance, best height wins
    idx = np.argmin(
        np.abs(df_peaks["center"].values[:, None] - ref_cm1[None, :]), axis=1
    )
    df_peaks = df_peaks.assign(ref_line=ref_cm1[idx])
    df_peaks = df_peaks.loc[
        (df_peaks["center"] - df_peaks["ref_line"]).abs() < NEON_MATCH_TOL_CM1
    ]
    df_peaks = (
        df_peaks.sort_values("height", ascending=False)
        .groupby("ref_line", as_index=False)
        .first()
    )
    return df_peaks.sort_values(by="center")


def fit_pixel_resolution_curve(centers, fwhms, degree=2):
    """CWA 18133 3.1.5: a function fit of FWHM vs neon peak position.

    Requires at least ``MIN_NEON_PEAKS`` points, and does one round of MAD-based
    outlier rejection so a single mis-fit neon peak does not distort the curve.
    Returns ``(poly, fit_lo, fit_hi)`` or ``(None, None, None)``."""
    centers = np.asarray(centers, dtype=float)
    fwhms = np.asarray(fwhms, dtype=float)
    if len(centers) < MIN_NEON_PEAKS:
        return None, None, None
    deg = min(degree, len(centers) - 1)
    if deg < 1:
        return None, None, None
    poly = np.poly1d(np.polyfit(centers, fwhms, deg))
    resid = fwhms - poly(centers)
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad > 0:
        keep = np.abs(resid - np.median(resid)) <= 3 * 1.4826 * mad
        if keep.sum() >= MIN_NEON_PEAKS and keep.sum() < len(centers):
            centers, fwhms = centers[keep], fwhms[keep]
            deg = min(degree, len(centers) - 1)
            poly = np.poly1d(np.polyfit(centers, fwhms, deg))
    return poly, float(centers.min()), float(centers.max())


def clip_to_range(x, lo, hi):
    """Boolean mask of ``x`` within ``[lo, hi]`` widened by ``CURVE_MARGIN_FRAC``."""
    margin = (hi - lo) * CURVE_MARGIN_FRAC
    return (x >= lo - margin) & (x <= hi + margin)


def fit_calcite_1085(spe_cal_calibrated, find_kw=None, prominence_coeff=3):
    """Voigt fit of the calcite ~1085.91 cm-1 peak (CWA Figure 2, Section 4).
    Falls back to Gaussian when the Voigt fit aborts (lmfit occasionally
    generates NaN model values); E2529 accepts mixed Gaussian/Lorentzian.
    Returns ``(peak_series_or_None, prepared_spectrum)``."""
    find_kw = dict(find_kw or DEFAULT_FIND_KW)
    spe = spe_cal_calibrated.dropna().trim_axes(
        method="x-axis",
        boundaries=(
            CALCITE_REF_CM1 - CALCITE_WINDOW_CM1,
            CALCITE_REF_CM1 + CALCITE_WINDOW_CM1,
        ),
    )
    spe.y = spe.y - np.min(spe.y)
    spe = spe.subtract_baseline_rc1_snip(niter=40)
    fitres = None
    profile = None
    for profile in ("Voigt", "Gaussian"):
        try:
            _find_kw = dict(find_kw)
            _find_kw["prominence"] = spe.y_noise_MAD() * prominence_coeff
            cand = spe.find_peak_multipeak(**_find_kw)
            fitres = spe.fit_peak_multimodel(
                profile=profile,
                candidates=cand,
                no_fit=False,
                bound_centers_to_group=True,
                vary_baseline=False,
            )
            break
        except Exception:  # noqa: BLE001 - try the next profile, then give up
            fitres = None
    if fitres is None:
        return None, spe
    df_peaks = fitres.to_dataframe_peaks()
    df_peaks = df_peaks.assign(profile=profile)
    df_peaks = df_peaks.loc[np.isfinite(df_peaks["fwhm"]) & (df_peaks["fwhm"] > 0)]
    if df_peaks.empty:
        return None, spe
    df_peaks = df_peaks.iloc[(df_peaks["center"] - CALCITE_REF_CM1).abs().argsort()]
    peak = df_peaks.iloc[0]
    if abs(peak["center"] - CALCITE_REF_CM1) > 20:
        return None, spe
    return peak, spe


def spectral_resolution_e2529(fwhm_1085, offset=E2529_OFFSET, slope=E2529_SLOPE):
    """ASTM E2529 spectral resolution: ``(FWHM_1085 - 0.684) / 1.0209``."""
    return (fwhm_1085 - offset) / slope


def _eval_clipped(curve, x_sped, in_range):
    """Evaluate a curve only within the neon-supported range (NaN elsewhere)."""
    if curve is None:
        return np.full(len(x_sped), np.nan)
    y = np.asarray(curve(x_sped), dtype=float)
    y[~in_range] = np.nan
    return y


@dataclass
class ResolutionResult:
    """CWA 18133 sections 3 & 4 products for one calibrated spectrum.

    Curves are on the calibrated Raman-shift grid ``raman_shift`` and are NaN
    outside the neon-supported span. ``spectral_res`` / ``sped_sres`` are all-NaN
    when there is no usable calcite fit; ``pixel_res`` is all-NaN when fewer than
    ``MIN_NEON_PEAKS`` neon peaks were fitted.
    """

    laser_wl: float
    raman_shift: np.ndarray
    sped: np.ndarray
    pixel_res: np.ndarray
    spectral_res: np.ndarray
    sped_sres: np.ndarray
    neon_peaks: pd.DataFrame
    n_neon_peaks: int
    curve_ok: bool
    curve_monotonic: bool | None
    fit_lo: float | None
    fit_hi: float | None
    neon_fwhm_median: float | None
    max_neon_fwhm_nm: float | None
    calcite_center: float | None
    calcite_fwhm: float | None
    calcite_profile: str | None
    spectral_resolution: float | None
    laser_effect_ratio: float | None
    uniform_grid: bool
    grid_step: float
    sres_plausible: bool | None
    within_cwa_boundary: bool | None
    notes: list[str] = field(default_factory=list)
    title: str = ""

    def figure(self):
        """The CWA Figure 2 three-panel plot (spectral distribution; pixel &
        spectral resolution; SpeD:SRes). Pure matplotlib -- returns the Figure."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        if self.title:
            fig.suptitle(self.title)

        ax1.plot(self.raman_shift, self.sped, color="#2a78d6")
        if self.uniform_grid:
            ax1.text(
                0.03,
                0.95,
                "resampled export grid,\nnot detector pixels",
                transform=ax1.transAxes,
                va="top",
                fontsize=8,
                color="#a33327",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        ax1.set_xlabel("Raman shift/cm⁻¹")
        ax1.set_ylabel("Spectral distribution/cm⁻¹ per pixel")
        ax1.set_title("Spectral distribution curve")
        ax1.grid(alpha=0.3)

        if not self.neon_peaks.empty:
            ax2.scatter(
                self.neon_peaks["center"],
                self.neon_peaks["fwhm"],
                label="Ne peak FWHM",
                color="#2a78d6",
            )
        if np.isfinite(self.pixel_res).any():
            ax2.plot(
                self.raman_shift,
                self.pixel_res,
                color="#eb6834",
                label="pixel resolution curve",
            )
        if np.isfinite(self.spectral_res).any():
            ax2.plot(
                self.raman_shift,
                self.spectral_res,
                color="#1baf7a",
                linestyle="--",
                label="spectral resolution curve",
            )
        if self.calcite_center is not None and self.spectral_resolution is not None:
            label = f"SRes (calcite, E2529) {self.spectral_resolution:.2f} cm⁻¹"
            if self.sres_plausible is False:
                label += " — implausible, not applied"
            ax2.scatter(
                [self.calcite_center],
                [self.spectral_resolution],
                color="#e34948",
                marker="x",
                s=80,
                label=label,
            )
        ax2.set_xlabel("Raman shift/cm⁻¹")
        ax2.set_ylabel("FWHM/cm⁻¹")
        ax2.set_title("Pixel & spectral resolution curves")
        ax2.grid(alpha=0.3)
        ax2.legend()

        if np.isfinite(self.sped_sres).any():
            ax3.plot(self.raman_shift, self.sped_sres, color="#4a3aa7")
        ax3.set_xlabel("Raman shift/cm⁻¹")
        ax3.set_ylabel("SpeD:SRes")
        ax3.set_title("SpeD:SRes curve")
        ax3.grid(alpha=0.3)

        fig.tight_layout()
        return fig


def resolution_from_calibration(
    calmodel: "CalibrationModel",
    spe_neon: Spectrum,
    *,
    neon_units: str = "cm-1",
    spe_calcite: Spectrum | None = None,
    calcite_units: str = "cm-1",
    find_kw=None,
    calcite_find_kw=None,
    prominence_coeff: float = 3,
    curve_fit_degree: int = 2,
    e2529_offset: float = E2529_OFFSET,
    e2529_slope: float = E2529_SLOPE,
    title: str = "",
) -> ResolutionResult:
    """CWA 18133 sections 3 & 4 for one x-calibration and its neon (+ calcite).

    ``calmodel`` is a fitted :class:`CalibrationModel`; ``spe_neon`` is the raw
    neon spectrum in ``neon_units``. The calibrated Raman-shift axis is obtained
    with ``calmodel.apply_calibration_x`` (x-calibration only). Pass ``spe_calcite``
    to add the ASTM E2529 spectral-resolution curve.
    """
    laser_wl = int(calmodel.laser_wl)
    notes: list[str] = [_SEAM_NOTE]

    # Section 3 - calibrated Raman shift axis applied to the neon spectrum
    uniform_grid, grid_step = detect_uniform_grid(spe_neon.x)
    if uniform_grid:
        notes.append(_UNIFORM_GRID_NOTE)
    spe_ne_cal = calmodel.apply_calibration_x(spe_neon, spe_units=neon_units).dropna()

    x_sped, sped = spectral_distribution(spe_ne_cal)
    ne_peaks = fit_neon_peaks(spe_ne_cal, laser_wl, find_kw, prominence_coeff)
    pixel_res_curve, fit_lo, fit_hi = fit_pixel_resolution_curve(
        ne_peaks["center"].values, ne_peaks["fwhm"].values, curve_fit_degree
    )
    in_range = (
        clip_to_range(x_sped, fit_lo, fit_hi)
        if pixel_res_curve is not None
        else np.zeros(len(x_sped), bool)
    )
    ne_peaks = ne_peaks.assign(
        fwhm_nm=[
            fwhm_cm1_to_nm(c, f, laser_wl)
            for c, f in zip(ne_peaks["center"], ne_peaks["fwhm"])
        ]
    )

    # Section 4 - calcite spectral resolution (ASTM E2529)
    calcite_peak, sres, ratio, sres_plausible = None, None, None, None
    if spe_calcite is not None and pixel_res_curve is not None:
        spe_cal_calibrated = calmodel.apply_calibration_x(
            spe_calcite, spe_units=calcite_units
        )
        calcite_peak, _ = fit_calcite_1085(
            spe_cal_calibrated, calcite_find_kw, prominence_coeff
        )
        if calcite_peak is not None:
            sres = spectral_resolution_e2529(
                calcite_peak["fwhm"], e2529_offset, e2529_slope
            )
            neon_fwhm_1085 = float(pixel_res_curve(calcite_peak["center"]))
            # laser effect adjustment: scale the pixel resolution curve so it
            # passes through the calcite spectral resolution value. A rescale
            # that pushes the curve well below the neon-derived instrument
            # function is physically impossible - the calcite fit is defective.
            _ratio = sres / neon_fwhm_1085 if neon_fwhm_1085 else 0.0
            if _ratio < SRES_MIN_RATIO or sres <= 0:
                sres_plausible = False
                notes.append(_IMPLAUSIBLE_NOTE)
            else:
                sres_plausible = True
                ratio = _ratio

    spectral_res_curve: Callable | None = (
        None if ratio is None else (lambda x, r=ratio, c=pixel_res_curve: r * c(x))
    )

    pixel_res = _eval_clipped(pixel_res_curve, x_sped, in_range)
    spectral_res = _eval_clipped(spectral_res_curve, x_sped, in_range)
    with np.errstate(divide="ignore", invalid="ignore"):
        sped_sres = sped / spectral_res
    sped_sres = np.where(np.isfinite(spectral_res), sped_sres, np.nan)

    curve_ok = pixel_res_curve is not None
    curve_monotonic = None
    if curve_ok:
        xr = np.linspace(fit_lo, fit_hi, 50)
        curve_monotonic = bool(np.all(np.diff(pixel_res_curve(xr)) >= -1e-9))
    max_fwhm_nm = ne_peaks["fwhm_nm"].max() if not ne_peaks.empty else np.nan

    return ResolutionResult(
        laser_wl=laser_wl,
        raman_shift=np.asarray(x_sped),
        sped=np.asarray(sped),
        pixel_res=pixel_res,
        spectral_res=spectral_res,
        sped_sres=sped_sres,
        neon_peaks=ne_peaks,
        n_neon_peaks=len(ne_peaks),
        curve_ok=curve_ok,
        curve_monotonic=curve_monotonic,
        fit_lo=fit_lo,
        fit_hi=fit_hi,
        neon_fwhm_median=(
            float(ne_peaks["fwhm"].median()) if not ne_peaks.empty else None
        ),
        max_neon_fwhm_nm=float(max_fwhm_nm) if np.isfinite(max_fwhm_nm) else None,
        calcite_center=None if calcite_peak is None else float(calcite_peak["center"]),
        calcite_fwhm=None if calcite_peak is None else float(calcite_peak["fwhm"]),
        calcite_profile=(None if calcite_peak is None else calcite_peak.get("profile")),
        spectral_resolution=None if sres is None else float(sres),
        laser_effect_ratio=None if ratio is None else float(ratio),
        uniform_grid=uniform_grid,
        grid_step=grid_step,
        sres_plausible=sres_plausible,
        within_cwa_boundary=(
            bool(max_fwhm_nm < PIXEL_RESOLUTION_BOUNDARY_NM)
            if np.isfinite(max_fwhm_nm)
            else None
        ),
        notes=notes,
        title=title,
    )


__all__ = [
    "ResolutionResult",
    "resolution_from_calibration",
    "spectral_distribution",
    "fit_neon_peaks",
    "fit_pixel_resolution_curve",
    "fit_calcite_1085",
    "spectral_resolution_e2529",
    "detect_uniform_grid",
    "neon_reference_cm1",
    "SRES_MIN_RATIO",
    "MIN_NEON_PEAKS",
]
