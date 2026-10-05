"""CWA 18133:2024 §8 portable calibration files.

§8 asks for a calibration file with metadata, date, the calibration curve as points
(uncalibrated shift -> calibrated shift), the silicon peak position and (optionally) the
calibrated laser wavelength; the curve points are meant to regenerate a spline. These
exporters write that as ``<base>.csv`` (the curve) + ``<base>.json`` (metadata AND the full
model in the portable :meth:`CalibrationModel.to_dict` form, so the exact model — not just
the sampled curve — can be reconstructed by ramanchada2 or any JSON-capable reader).

``export_nexus_calibration`` writes the same content, plus the calibrant spectra it was
derived from, as a self-describing NXraman HDF5 file (plain h5py or h5pyd).
"""
import datetime
import json

import numpy as np

from ramanchada2.spectrum import Spectrum

SI_REF_CM1 = 520.45


def _calibration_curve(calmodel, spectral_range, npoints):
    """Sample the model over ``spectral_range`` (cm-1): uncalibrated -> calibrated."""
    lo, hi = float(min(spectral_range)), float(max(spectral_range))
    grid = np.linspace(lo, hi, int(npoints))  # endpoints exactly at the range bounds (§8)
    spe = Spectrum(x=grid, y=np.ones_like(grid))
    calibrated = calmodel.apply_calibration_x(spe, spe_units="cm-1").x
    return grid, np.asarray(calibrated, dtype=float)


def _laser_zero_info(calmodel):
    """Si peak position (nm and implied cm-1 on the nominal axis) and calibrated laser wl."""
    from .xcalibration import LazerZeroingComponent
    for c in calmodel.components:
        if isinstance(c, LazerZeroingComponent):
            zero_nm = float(c.model)
            si_ref = float(list(c.ref.keys())[0]) if c.ref else SI_REF_CM1
            laser_nm = 1e7 / (1e7 / zero_nm + si_ref)
            return {
                "si_peak_nm": zero_nm,
                "si_reference_cm1": si_ref,
                "calibrated_laser_wl_nm": laser_nm,
            }
    return {}


def export_cwa_x(calmodel, base_path, spectral_range, npoints=200, metadata=None):
    """Write ``<base_path>.csv`` + ``<base_path>.json`` for an x-calibration model.

    Args:
        calmodel: a derived :class:`CalibrationModel` (Ne curve + Si zeroing components).
        base_path: output path without extension.
        spectral_range: (min, max) Raman shift in cm-1 of the spectra this calibration is
            intended for; the CSV endpoints correspond exactly to these bounds (§8).
        npoints: number of curve points.
        metadata: optional dict merged into the JSON (instrument metadata per CWA §5.3.3).
    """
    grid, calibrated = _calibration_curve(calmodel, spectral_range, npoints)
    csv_path = f"{base_path}.csv"
    json_path = f"{base_path}.json"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("uncalibrated_cm1,calibrated_cm1\n")
        for u, c in zip(grid, calibrated):
            f.write(f"{u:.6f},{c:.6f}\n")

    doc = {
        "format": "CWA18133-x-calibration",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "laser_wl_nominal_nm": int(calmodel.laser_wl) if calmodel.laser_wl is not None else None,
        **_laser_zero_info(calmodel),
        "spectral_range_cm1": [float(min(spectral_range)), float(max(spectral_range))],
        "curve_csv": csv_path.replace("\\", "/").split("/")[-1],
        "metadata": metadata or {},
        "model": calmodel.to_dict(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return csv_path, json_path


def export_cwa_y(ycal_component, base_path, spectral_range=None, npoints=200,
                 metadata=None, x_calibration_ref=None):
    """Write ``<base_path>.csv`` + ``<base_path>.json`` for a y-calibration component.

    The CSV holds intensity factors as a function of the calibrated Raman shift (§8):
    factor(x) = certificate_response(x) / measured_reference(x), masked where the measured
    reference is at/below its noise (same rule as YCalibrationComponent.safe_factor).
    """
    cert = ycal_component.ref
    if spectral_range is None:
        spectral_range = cert.raman_shift
    lo, hi = float(min(spectral_range)), float(max(spectral_range))
    grid = np.linspace(lo, hi, int(npoints))
    measured = np.asarray(ycal_component.model(grid), dtype=float)
    expected = np.asarray(cert.Y(grid), dtype=float)
    factor = np.zeros_like(grid)
    ref_noise = 0.0  # dense evaluation of the smoothed model; mask only non-positive values
    mask = (measured > ref_noise) & np.isfinite(expected)
    factor[mask] = expected[mask] / measured[mask]

    csv_path = f"{base_path}.csv"
    json_path = f"{base_path}.json"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("calibrated_cm1,intensity_factor\n")
        for x, y in zip(grid, factor):
            f.write(f"{x:.6f},{y:.8g}\n")

    doc = {
        "format": "CWA18133-y-calibration",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "laser_wl_nominal_nm": int(ycal_component.laser_wl) if ycal_component.laser_wl is not None else None,
        "certificate": cert.model_dump(),
        "x_calibration_ref": x_calibration_ref,
        "spectral_range_cm1": [lo, hi],
        "curve_csv": csv_path.replace("\\", "/").split("/")[-1],
        "metadata": metadata or {},
        "model": ycal_component.to_dict(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return csv_path, json_path


def _string_dtype(h5):
    """Variable-length utf-8 string dtype for ``h5`` (h5py or h5pyd)."""
    if hasattr(h5, "string_dtype"):
        return h5.string_dtype()
    return h5.special_dtype(vlen=str)


def _h5_clean(value, h5):
    """Coerce a to_dict()-sourced value into something create_dataset accepts.

    Plain numpy string arrays (numpy's default fixed-width unicode dtype, e.g. '<U8')
    have no direct HDF5 type mapping and need a variable-length utf-8 dtype. Numeric
    lists/tuples become plain float arrays. Everything else (scalar numbers, bools,
    already-clean strings) passes through unchanged.
    """
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, str) for v in value):
            return np.asarray(value, dtype=_string_dtype(h5))
        return np.asarray(value, dtype=float)
    return value


def _component_parameters(component_dict):
    """Extract the reconstructable fit parameters from one to_dict() component payload,
    for NXcalibration/calibration_parameters — separate from the anchors (matched peaks,
    a different length) and from calibration_object (the full JSON, kept verbatim).

    Returns a flat ``{name: value}`` dict of JSON-scalar/array values suitable for writing
    as sibling datasets in an NXparameters group. Shape depends on the interpolator/
    component type; unknown shapes fall back to an empty dict rather than raising, since
    calibration_object already carries the complete, authoritative payload.
    """
    model = component_dict.get("model")
    out = {}
    if component_dict.get("type") == "LazerZeroingComponent":
        # model is a plain float: the fitted Si peak position, nm
        out["si_peak_nm"] = float(model)
        return out
    if not isinstance(model, dict):
        return out
    kind = model.get("type")
    if kind == "CustomPolyInterpolator":
        for key in ("coef", "degree", "x_min", "x_max", "fit_error"):
            if key in model and model[key] is not None:
                out[key] = model[key]
    elif kind in ("CustomPChipInterpolator", "CustomCubicSplineInterpolator"):
        if "x" in model:
            out["knots_x"] = model["x"]
        if "y" in model:
            out["knots_y"] = model["y"]
    elif kind == "ParametricModel":
        out["equation"] = model.get("equation")
        out["param_names"] = model.get("param_names")
        out["coef"] = model.get("coef")
    return out


def _fit_formula_description(component_dict):
    """Human-readable summary of the interpolator + its extrapolation rule.

    Both CustomPolyInterpolator and CustomPChipInterpolator continue with CONSTANT
    CORRECTION (unit slope) beyond the anchor span rather than an unconstrained tail
    (see interpolators.py) — a reader that reimplements naive extrapolation will diverge
    from this model by tens of nm outside the anchor range, so the rule is stated here.
    """
    if component_dict.get("type") == "LazerZeroingComponent":
        profile = component_dict.get("profile", "Pearson4")
        return f"Silicon 520.45 cm-1 peak position, fitted with a {profile} profile."
    model = component_dict.get("model")
    kind = model.get("type") if isinstance(model, dict) else None
    extrapolation = ("Beyond the anchor span the map continues with constant correction "
                     "(unit slope) from the edge value, not raw polynomial/spline "
                     "extrapolation.")
    if kind == "CustomPolyInterpolator":
        degree = model.get("degree")
        return (f"Polynomial degree {degree}, fit on x normalized to "
                f"u=(x-x_min)/(x_max-x_min); calibrated = polyval(coef, u). "
                f"{extrapolation}")
    if kind == "CustomPChipInterpolator":
        return f"Monotone PCHIP through (knots_x, knots_y). {extrapolation}"
    if kind == "CustomCubicSplineInterpolator":
        bc = model.get("bc_type") if isinstance(model, dict) else None
        return f"Cubic spline (bc_type={bc}) through (knots_x, knots_y). {extrapolation}"
    if kind == "ParametricModel":
        return f"Analytic model: {model.get('equation')}"
    return kind or "unknown"


def _axis_name_for_units(units):
    return {"cm-1": "raman_shift", "nm": "wavelength", "pixel": "pixel"}.get(units, "x")


# component_dict keys that already have a typed, *numeric* home elsewhere in the written
# NXcalibration group (original_axis/calibrated_axis, anchors) or that are orchestration-
# level, not calibration-model content (name/enabled, handled via attrs already).
# "model" and "certificate" are deliberately NOT here, even though _component_parameters
# also surfaces a flattened numeric view of "model" under calibration_parameters/: that
# flattened view is a convenience mirror for readers that only want the numbers, not a
# substitute for the exact tagged-dict shape interpolator_from_tagged_dict/
# YCalibrationComponent.from_dict need. Residual field pruning must never remove
# anything the corresponding from_dict() classmethod requires.
_TYPED_ELSEWHERE = {
    "anchors", "name", "enabled", "sample", "spe_units", "ref_units",
    "model_units", "match_method", "interpolator_method", "ref", "profile",
}


def _residual_component_fields(component_dict):
    """component_dict stripped of everything that already has a typed home (see
    _TYPED_ELSEWHERE), so calibration_object/NXnote carries only what genuinely has none
    yet -- e.g. laser_wl, nonmonotonic policy, extrapolate flag, model_method."""
    return {k: v for k, v in component_dict.items() if k not in _TYPED_ELSEWHERE}


def _certificate_curve(cert, grid):
    """Sample a YCalibrationCertificate's analytic response over ``grid`` (calibrated
    cm-1), for a plottable certificate-response NXdata alongside the measured SRM
    spectrum it was compared against."""
    return np.asarray(cert.Y(grid), dtype=float)


def _nx_group(parent, name, nx_class, **attrs):
    group = parent.require_group(name)
    group.attrs["NX_class"] = nx_class
    for key, value in attrs.items():
        group.attrs[key] = value
    return group


def _nx_data(parent, name, signal, axes, arrays, units=None, **attrs):
    """Write a plottable NXdata group: ``arrays`` is an ordered ``{dataset: values}``
    dict; ``signal`` names the signal dataset and ``axes`` the axis datasets."""
    group = _nx_group(parent, name, "NXdata", signal=signal, axes=list(axes),
                      interpretation="spectrum", **attrs)
    for key, values in arrays.items():
        ds = group.create_dataset(key, data=np.asarray(values, dtype=float))
        if units and key in units:
            ds.attrs["units"] = units[key]
    return group


def _nx_note(parent, name, document, h5):
    note = _nx_group(parent, name, "NXnote")
    note.create_dataset("type", data="application/json")
    note.create_dataset("data", data=json.dumps(document))
    return note


def _nx_calibration_group(parent, name, component_dict, h5, curve=None):
    """Write one calibration component as an NXcalibration group under ``parent``.

    ``curve`` is an optional (original_axis, calibrated_axis) array pair — real datasets.
    calibration_parameters is an NXparameters container with the actual numeric children;
    calibration_object/NXnote is kept deliberately small — only fields with no typed home
    yet (see _residual_component_fields) — so the typed datasets are the single source of
    truth for the reconstructable numbers rather than a duplicate copy.
    """
    is_y = component_dict.get("type") == "YCalibrationComponent"
    cal = _nx_group(
        parent, name, "NXcalibration",
        description="ramanchada2 " + component_dict.get("type", "calibration"),
        physical_quantity="relative intensity" if is_y else "wavenumber",
        applied=bool(component_dict.get("enabled", True)),
        fit_formula_description=_fit_formula_description(component_dict),
    )

    if curve is not None:
        original_axis, calibrated_axis = curve
        cal.create_dataset("original_axis", data=np.asarray(original_axis, dtype=float))
        cal.create_dataset("calibrated_axis", data=np.asarray(calibrated_axis, dtype=float))

    params = _component_parameters(component_dict)
    if params:
        pgroup = _nx_group(cal, "calibration_parameters", "NXparameters")
        for key, value in params.items():
            if value is None:
                continue
            pgroup.create_dataset(key, data=_h5_clean(value, h5))

    _nx_note(cal, "calibration_object", _residual_component_fields(component_dict), h5)

    anchors = component_dict.get("anchors")
    if anchors:
        arr = np.asarray(anchors, dtype=float)
        agroup = _nx_group(
            cal, "anchors", "NXdata",
            description="Matched calibrant peaks used to derive the model.")
        agroup.create_dataset("measured", data=arr[:, 0])
        agroup.create_dataset("reference", data=arr[:, 1])
        agroup.create_dataset("inlier", data=arr[:, 2].astype(bool))
    return cal


def export_nexus_calibration(
    calmodel,
    filename,
    spectral_range=(100, 3500),
    npoints=200,
    metadata=None,
    instrument=None,
    ycal_component=None,
    spe_neon=None,
    spe_neon_units="cm-1",
    spe_silicon=None,
    spe_silicon_units="cm-1",
    title=None,
    wavelength=None,
    h5module=None,
):
    """Write the calibration workflow as a self-describing NeXus (NXraman) HDF5 file.

    Only ``h5py`` (or ``h5pyd`` via ``h5module``, as in :func:`ramanchada2.io.HSDS.write_nexus`)
    is used. The entry carries ``definition = "NXraman"``, an ``NXinstrument`` (device
    information, incident beam wavelength, one ``NXcalibration`` group per calibration
    component), an ``NXsample`` and plottable ``NXdata`` groups for the calibrant spectra
    actually used (``spe_neon``/``spe_silicon``, as loaded), the sampled x calibration curve
    and — with ``ycal_component`` — the measured SRM response and its certificate curve.

    The full model is stored as JSON in ``entry/calibration_model/data``, so a reader can
    regenerate ``calmodel`` from the file alone via
    ``CalibrationModel.from_dict(json.loads(...)["model"])``.

    Args:
        calmodel: a derived :class:`CalibrationModel` (Ne curve + Si zeroing components).
        filename: output ``.nxs``/``.h5`` path (or HSDS domain for ``h5pyd``).
        spectral_range: (min, max) Raman shift in cm-1 the sampled curve should cover.
        npoints: number of curve points.
        metadata: optional dict stored in the JSON model document.
        instrument: optional dict of instrument metadata. ``instrument_make``/
            ``instrument_model`` become ``NXinstrument/device_information`` (vendor,
            model); ``laser_wl`` is used for the beam wavelength if ``wavelength`` is not
            given; other keys are written as datasets under ``instrument/parameters``.
            NaN/None values are dropped.
        ycal_component: optional derived ``YCalibrationComponent`` (relative-intensity
            calibration); written as a second NXcalibration group together with its
            measured SRM spectrum and certificate response curve.
        spe_neon, spe_silicon: optional as-loaded calibrant :class:`Spectrum` objects (the
            *inputs* the model was derived from). When omitted, no reference spectra are
            written.
        spe_neon_units, spe_silicon_units: axis units of the calibrant spectra as loaded
            ("cm-1", "nm", or "pixel") — do not assume cm-1.
        title: optional entry title.
        wavelength: incident laser wavelength in nm.
        h5module: ``h5py`` (default) or ``h5pyd``.

    Returns:
        filename
    """
    if h5module is None:
        import h5py as h5module
    h5 = h5module

    grid, calibrated = _calibration_curve(calmodel, spectral_range, npoints)

    def _not_missing(value):
        return value is not None and not (isinstance(value, float) and np.isnan(value))

    instrument = {k: v for k, v in (instrument or {}).items() if _not_missing(v)}
    if wavelength is None:
        try:
            wavelength = float(instrument.get("laser_wl"))
        except (TypeError, ValueError):
            wavelength = None

    spectra = []  # (group name, x, y, x_units, x_name, y_name, description)
    for name, spe, units in (("reference_neon", spe_neon, spe_neon_units),
                             ("reference_silicon", spe_silicon, spe_silicon_units)):
        if spe is not None:
            spectra.append((name, spe.x, spe.y, units, "intensity", "As-loaded calibrant."))

    y_curves = []
    if ycal_component is not None:
        y_grid = grid
        cert = getattr(ycal_component, "ref", None)
        if cert is not None and getattr(cert, "raman_shift", None):
            lo, hi = cert.raman_shift
            y_grid = np.linspace(float(lo), float(hi), int(npoints))
        try:
            measured = np.asarray(ycal_component.model(y_grid), dtype=float)
            y_curves.append(("calibration_curve_y_measured", y_grid, measured,
                             "Measured SRM response, resampled onto the certificate's "
                             "declared range."))
        except Exception:
            pass
        if cert is not None:
            try:
                y_curves.append(("calibration_curve_y_certificate", y_grid,
                                 _certificate_curve(cert, y_grid),
                                 "Certificate response curve."))
            except Exception:
                pass

    with h5.File(filename, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = "entry"
        entry = _nx_group(f, "entry", "NXentry", default="calibration_curve")
        entry.create_dataset("definition", data="NXraman")
        entry.create_dataset("title", data=title or "ramanchada2 x/y calibration")
        entry.create_dataset("experiment_type", data="Raman spectroscopy")

        inst = _nx_group(entry, "instrument", "NXinstrument")
        device = _nx_group(inst, "device_information", "NXfabrication")
        device.create_dataset("vendor", data=str(instrument.get("instrument_make") or "unknown"))
        device.create_dataset("model", data=str(instrument.get("instrument_model") or "unknown"))
        if wavelength is not None:
            beam = _nx_group(inst, "beam_incident", "NXbeam")
            wl = beam.create_dataset("wavelength", data=float(wavelength))
            wl.attrs["units"] = "nm"
        extras = {k: v for k, v in instrument.items()
                  if k not in ("instrument_make", "instrument_model", "laser_wl")}
        if extras:
            params = _nx_group(inst, "parameters", "NXparameters")
            for key, value in extras.items():
                params.create_dataset(key, data=value)

        sample = _nx_group(entry, "sample", "NXsample")
        sample.create_dataset("name", data=title or "calibration")

        for name, x, y, units, y_name, description in spectra:
            x_name = _axis_name_for_units(units)
            _nx_data(entry, name, y_name, [x_name], {x_name: x, y_name: y},
                     units={x_name: units}, description=description)

        doc = {"metadata": metadata or {}, **_laser_zero_info(calmodel),
               "model": calmodel.to_dict()}
        _nx_note(entry, "calibration_model", doc, h5)

        _nx_data(entry, "calibration_curve", "calibrated_cm1", ["uncalibrated_cm1"],
                 {"uncalibrated_cm1": grid, "calibrated_cm1": calibrated},
                 units={"uncalibrated_cm1": "1/cm", "calibrated_cm1": "1/cm"})

        x_components = [c for c in getattr(calmodel, "components", [])
                        if c.to_dict().get("type") != "YCalibrationComponent"]
        for idx, comp in enumerate(x_components):
            name = f"calibration_x_{idx}" if len(x_components) > 1 else "calibration_x"
            _nx_calibration_group(inst, name, comp.to_dict(), h5,
                                  curve=(grid, calibrated) if idx == 0 else None)

        if ycal_component is not None:
            _nx_calibration_group(inst, "calibration_y", ycal_component.to_dict(), h5)
            for name, x, y, description in y_curves:
                if name == "calibration_curve_y_measured":
                    # plottable y-calibration curve, the analogue of calibration_curve
                    _nx_data(entry, "calibration_curve_y", "intensity_factor",
                             ["calibrated_cm1"], {"calibrated_cm1": x, "intensity_factor": y},
                             description=description)
                _nx_data(entry, name, "intensity_factor", ["calibrated_cm1"],
                         {"calibrated_cm1": x, "intensity_factor": y},
                         description=description)
    return filename
