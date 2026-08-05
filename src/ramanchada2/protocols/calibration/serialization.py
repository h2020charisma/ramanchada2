"""CWA 18133:2024 §8 portable calibration files.

§8 asks for a calibration file with metadata, date, the calibration curve as points
(uncalibrated shift -> calibrated shift), the silicon peak position and (optionally) the
calibrated laser wavelength; the curve points are meant to regenerate a spline. These
exporters write that as ``<base>.csv`` (the curve) + ``<base>.json`` (metadata AND the full
model in the portable :meth:`CalibrationModel.to_dict` form, so the exact model — not just
the sampled curve — can be reconstructed by ramanchada2 or any JSON-capable reader).

``export_nexus_calibration`` writes the same content, plus the calibrant spectra it was
derived from, as a self-describing NeXus/HDF5 file (NXentry/NXinstrument/NXcalibration).
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


def _h5_clean(value):
    """Coerce a to_dict()-sourced value into something h5py's create_dataset accepts.

    Plain numpy string arrays (numpy's default fixed-width unicode dtype, e.g. '<U8')
    have no direct HDF5 type mapping; h5py needs its own variable-length utf-8 dtype for
    string data. Numeric lists/tuples become plain float arrays. Everything else (scalar
    numbers, bools, already-clean strings) passes through unchanged.
    """
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, str) for v in value):
            return np.asarray(value, dtype=h5py_string_dtype())
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        return value
    return value


def h5py_string_dtype():
    import h5py
    return h5py.special_dtype(vlen=str)


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
# level, not calibration-model content (name/enabled, handled via NX_class attrs already).
# "model" and "certificate" are deliberately NOT here, even though _component_parameters
# also surfaces a flattened numeric view of "model" under calibration_parameters/: that
# flattened view is a convenience mirror for readers that only want the numbers, not a
# substitute for the exact tagged-dict shape interpolator_from_tagged_dict/
# YCalibrationComponent.from_dict need (e.g. ParametricModel needs "type"+"equation"+
# "param_names"+"coef" together; _component_parameters doesn't reproduce every
# interpolator kind's exact shape). Dropping "model" broke the ycal_component per-
# component round-trip in exactly this way (KeyError: 'model') -- residual field pruning
# must never remove anything the corresponding from_dict() classmethod requires.
_TYPED_ELSEWHERE = {
    "anchors", "name", "enabled", "sample", "spe_units", "ref_units",
    "model_units", "match_method", "interpolator_method", "ref", "profile",
}


def _residual_component_fields(component_dict):
    """component_dict stripped of everything that already has a typed home (see
    _TYPED_ELSEWHERE), so calibration_object/NXnote carries only what genuinely has none
    yet -- e.g. laser_wl, nonmonotonic policy, extrapolate flag, model_method -- rather
    than the full model duplicated verbatim alongside the typed datasets."""
    return {k: v for k, v in component_dict.items() if k not in _TYPED_ELSEWHERE}


def _build_measurement_papp(spe_list, meta, instrument=None, wavelength=None,
                            provider="ramanchada2", investigation="calibration",
                            sample="calibration"):
    """Build one NXRamanProtocolApplication carrying every (x, y, endpointtype, nx_name,
    units) tuple in ``spe_list`` as its own EffectArray, via pyambit's spe2ambit/
    configure_papp — the same machinery every other Raman measurement in this ecosystem
    goes through (see pyambit.nexus_spectra), rather than a hand-rolled NXdata group, so
    the written file gets real NXRaman instrument/sample linkage and the same plottable
    signal/axes/interpretation conventions process_pa already implements.

    pyambit is an optional dependency of ramanchada2 (not a hard one — most rc2 consumers
    have no need for it), so only the *absence* of pyambit falls back silently (returns
    None; export_nexus_calibration then writes a minimal plottable NXdata fallback with no
    NXRaman typing). A pyambit call that raises for any other reason is a real bug in the
    export path and must not be swallowed here — it propagates.
    """
    try:
        from pyambit.nexus_spectra import spe2ambit
    except ImportError:
        return None

    # spe2ambit/configure_papp unconditionally subscript instrument[0]/instrument[1]
    # (vendor, model) with no None-guard, so a default of None crashes on the very first
    # call rather than degrading gracefully -- always pass a concrete (possibly unknown)
    # pair instead of relying on pyambit to handle the missing-metadata case.
    instrument = instrument if instrument is not None else ("unknown", "unknown")

    papp = None
    for x, y, endpointtype, nx_name, units in spe_list:
        # spe2ambit's `sample` argument does double duty: on the FIRST call (papp is
        # None) it also seeds configure_papp's overall Sample.uuid identity for the whole
        # papp; on every call it becomes spe2effect's nx_name -- the per-effect group name
        # process_pa uses to build entryid ("{nx_name}_{index}"). Passing the per-spectrum
        # nx_name here (not the overall `sample` parameter) on every call is deliberate:
        # it makes each effect's group discoverable by its own name (reference_neon,
        # calibration_curve_x, ...) instead of every effect sharing one generic label,
        # which is what made the previous version's groups indistinguishable except by
        # positional index (RAW_DATA/calibration_1, RAW_DATA/calibration_2, ...).
        # meta must carry BOTH the per-effect @signal/@axes routing keys AND the
        # caller-supplied meta (grating, slit_size, ...) -- configure_papp only reads
        # meta on the FIRST spe2ambit call (papp is None), so if that first call's meta
        # doesn't include the caller's instrument metadata, it is never seen again for
        # the rest of this papp's lifetime. An earlier version passed only the routing
        # keys here, silently dropping every meta-sourced field except wavelength/
        # instrument (those come from configure_papp's own dedicated parameters, not
        # from meta).
        effect_meta = dict(meta or {})
        effect_meta["@signal"] = "y"
        effect_meta["@axes"] = [_axis_name_for_units(units)]
        papp = spe2ambit(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float),
            meta=effect_meta,
            instrument=instrument, wavelength=wavelength, provider=provider,
            investigation=investigation, sample=nx_name, endpointtype=endpointtype,
            unit=units, papp=papp,
        )
    papp.sync_parameters()
    return papp


def _certificate_curve(cert, grid):
    """Sample a YCalibrationCertificate's analytic response over ``grid`` (calibrated
    cm-1), for a plottable certificate-response NXdata alongside the measured SRM
    spectrum it was compared against."""
    return np.asarray(cert.Y(grid), dtype=float)


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
    provider="ramanchada2",
    investigation="calibration",
):
    """Write the calibration workflow as a self-describing NeXus/HDF5 file.

    Carries, in reconstructable form: the calibrant spectra actually used
    (``spe_neon``/``spe_silicon``, as-loaded, plus — when ``ycal_component`` is given —
    the measured SRM spectrum and its certificate's response curve), the sampled x and y
    calibration curves, and — inside an ``NXcalibration`` group per component — the
    portable fit parameters (curve points, interpolator knots or polynomial coefficients,
    Si zeroing, matched-peak anchors). A reader can regenerate ``calmodel`` from the file
    alone via ``CalibrationModel.from_dict(json.loads(.../entry/calibration_model/data))``
    without needing pyambit or ramanchada2 installed to parse the file structure itself.

    Every spectrum (calibrants, sampled curves, certificate response) is written through
    pyambit's ``spe2ambit``/``NXRamanProtocolApplication`` machinery when pyambit is
    importable, so it gets real NXRaman instrument/sample linkage and the same plottable
    ``signal``/``axes``/``interpretation`` conventions as any other Raman measurement in
    this ecosystem — not a hand-rolled, one-off NXdata shape. If pyambit is not installed,
    a minimal plottable NXdata fallback (same NIAC signal/axes attributes, no NXRaman
    typing) is written instead, so the export never fails outright over an optional
    dependency; the ``NXcalibration``-specific structures (fit parameters, anchors) always
    go through plain h5py/nexusformat, since neither pyambit's codegen (scalar-typed
    array fields, see docs/nexus_export_plan.md) nor NXRaman itself has a home for them.

    Args:
        calmodel: a derived :class:`CalibrationModel` (Ne curve + Si zeroing components).
        filename: output ``.nxs``/``.h5`` path.
        spectral_range: (min, max) Raman shift in cm-1 the sampled curve should cover.
        npoints: number of curve points.
        metadata: optional dict merged into the top-level entry (instrument/run metadata).
        instrument: optional dict of instrument metadata. ``instrument_make``/
            ``instrument_model`` (if present) become pyambit's ``(vendor, model)`` device
            identity; every other key (``grating``, ``slit_size``, ...) is routed through
            pyambit's ``configure_papp`` backward-compat key table / generic parameters
            bucket, same as any other Raman measurement's meta dict.
        ycal_component: optional derived ``YCalibrationComponent`` (relative-intensity
            calibration); written as a second NXcalibration group when given, together
            with its measured SRM spectrum and certificate response curve.
        spe_neon, spe_silicon: optional as-loaded calibrant :class:`Spectrum` objects (the
            *inputs* the model was derived from, before any trim/baseline preprocessing —
            that preprocessing is derivable from these; the reverse is not). When omitted,
            no reference spectra are written (they are optional, unlike the model itself).
        spe_neon_units, spe_silicon_units: axis units of the respective calibrant spectra
            as loaded ("cm-1", "nm", or "pixel") — do not assume cm-1.
        title: optional entry title.
        wavelength, provider, investigation: forwarded to pyambit's configure_papp for
            the NXRaman instrument/citation context (see pyambit.nexus_spectra).

    Returns:
        filename
    """
    grid, calibrated = _calibration_curve(calmodel, spectral_range, npoints)

    spe_list = []
    if spe_neon is not None:
        spe_list.append((spe_neon.x, spe_neon.y, "RAW_DATA", "reference_neon", spe_neon_units))
    if spe_silicon is not None:
        spe_list.append(
            (spe_silicon.x, spe_silicon.y, "RAW_DATA", "reference_silicon", spe_silicon_units))
    spe_list.append((grid, calibrated, "X_CALIBRATION", "calibration_curve_x", "cm-1"))

    if ycal_component is not None:
        y_grid = grid
        cert = getattr(ycal_component, "ref", None)
        if cert is not None and getattr(cert, "raman_shift", None):
            lo, hi = cert.raman_shift
            y_grid = np.linspace(float(lo), float(hi), int(npoints))
        try:
            measured = np.asarray(ycal_component.model(y_grid), dtype=float)
            spe_list.append(
                (y_grid, measured, "Y_CALIBRATION_MEASURED", "calibration_curve_y_measured",
                 "cm-1"))
        except Exception:
            pass
        if cert is not None:
            try:
                cert_curve = _certificate_curve(cert, y_grid)
                spe_list.append(
                    (y_grid, cert_curve, "Y_CALIBRATION_CERTIFICATE",
                     "calibration_curve_y_certificate", "cm-1"))
            except Exception:
                pass

    def _not_missing(value):
        return value is not None and not (isinstance(value, float) and np.isnan(value))

    instrument_tuple = None
    meta = {k: v for k, v in (metadata or {}).items() if _not_missing(v)}
    if instrument:
        instrument_tuple = (
            instrument.get("instrument_make") or "unknown",
            instrument.get("instrument_model") or "unknown",
        )
        meta.update({k: v for k, v in instrument.items()
                    if k not in ("instrument_make", "instrument_model") and _not_missing(v)})

    papp = _build_measurement_papp(
        spe_list, meta=meta, instrument=instrument_tuple, wavelength=wavelength,
        provider=provider, investigation=investigation,
        sample=(title or "ramanchada2 x/y calibration"))

    if papp is not None:
        import nexusformat.nexus.tree as nx

        nx_root = nx.NXroot()
        # papp.to_nexus(nx_root) (the bound method), not the free
        # pyambit.nexus_writer.to_nexus(papp, nx_root=...) function: to_nexus writes the
        # entry at a leading-slash path derived from (provider, papp.uuid), and
        # nx_root.entries (not .keys()) is what correctly resolves that -- the pattern
        # pyambit's own test suite uses (tests/pyambit/nexus_models/nx_raman_test.py).
        papp.to_nexus(nx_root)
        entry = next(iter(nx_root.entries.values()))
        if "instrument" not in entry:
            entry["instrument"] = nx.NXinstrument()
        inst = entry["instrument"]
    else:
        # pyambit not importable: minimal plottable fallback, still real NXdata with the
        # documented signal/axes conventions, just not NXRaman-typed.
        import nexusformat.nexus.tree as nx

        nx_root = nx.NXroot()
        entry = nx.NXentry()
        entry["title"] = title or "ramanchada2 x/y calibration"
        entry["instrument"] = nx.NXinstrument()
        inst = entry["instrument"]
        for name, x, y, x_units, y_name in (
            ("reference_neon", spe_neon.x if spe_neon is not None else None,
             spe_neon.y if spe_neon is not None else None, spe_neon_units, "intensity"),
            ("reference_silicon", spe_silicon.x if spe_silicon is not None else None,
             spe_silicon.y if spe_silicon is not None else None, spe_silicon_units,
             "intensity"),
        ):
            if x is None:
                continue
            data = nx.NXdata()
            x_name = _axis_name_for_units(x_units)
            data[x_name] = np.asarray(x, dtype=float)
            data[y_name] = np.asarray(y, dtype=float)
            data.attrs["signal"] = y_name
            data.attrs["axes"] = [x_name]
            entry[name] = data
        nx_root["entry"] = entry

    entry["calibration_model"] = nx.NXnote()
    entry["calibration_model"]["type"] = "application/json"
    doc = {"metadata": metadata or {}, **_laser_zero_info(calmodel), "model": calmodel.to_dict()}
    entry["calibration_model"]["data"] = json.dumps(doc)

    curve_data = nx.NXdata()
    curve_data["uncalibrated_cm1"] = np.asarray(grid, dtype=float)
    curve_data["calibrated_cm1"] = np.asarray(calibrated, dtype=float)
    curve_data.attrs["signal"] = "calibrated_cm1"
    curve_data.attrs["axes"] = ["uncalibrated_cm1"]
    entry["calibration_curve"] = curve_data
    entry.attrs["default"] = "calibration_curve"

    components = list(getattr(calmodel, "components", []))
    x_components = [c for c in components if c.to_dict().get("type") != "YCalibrationComponent"]
    for idx, comp in enumerate(x_components):
        comp_dict = comp.to_dict()
        curve = (grid, calibrated) if idx == 0 else None
        name = f"calibration_x_{idx}" if len(x_components) > 1 else "calibration_x"
        inst[name] = _nx_calibration_group(comp_dict, curve=curve)

    if ycal_component is not None:
        y_dict = ycal_component.to_dict()
        inst["calibration_y"] = _nx_calibration_group(y_dict)
        # Plottable y-calibration curve, the analogue of calibration_curve for x -- the
        # gap that made the earlier h5py-only version look like intensity calibration was
        # silently dropped (it wasn't; it just had no plottable representation).
        if "calibration_curve_y_measured" in (spe_names := [s[3] for s in spe_list]):
            y_curve = nx.NXdata()
            idx = spe_names.index("calibration_curve_y_measured")
            y_curve["calibrated_cm1"] = np.asarray(spe_list[idx][0], dtype=float)
            y_curve["intensity_factor"] = np.asarray(spe_list[idx][1], dtype=float)
            y_curve.attrs["signal"] = "intensity_factor"
            y_curve.attrs["axes"] = ["calibrated_cm1"]
            y_curve.attrs["description"] = ("Measured SRM response, resampled onto the "
                                            "certificate's declared range.")
            entry["calibration_curve_y"] = y_curve

    nx_root.save(filename, mode="w")
    return filename


def _nx_calibration_group(component_dict, curve=None):
    """Build one calibration component as an NXcalibration nexusformat.nexus.tree group,
    for composing into the shared NXroot alongside the pyambit-written entry (the whole
    file is saved in one nx_root.save() call — see export_nexus_calibration). ``curve``
    is an optional (original_axis, calibrated_axis) array pair — real datasets, not the
    scalar-typed fields pyambit's generated NXCalibration pydantic model exposes (that
    codegen has no NX_FLOAT[rank] support yet; see docs/nexus_export_plan.md).
    calibration_parameters is a real NXparameters container with its actual numeric
    children, not the near-empty base-class stub; calibration_object/NXnote is kept
    deliberately small — only fields with no typed home yet (see
    _residual_component_fields) — so the typed datasets are the single source of truth
    for the reconstructable numbers rather than a duplicate copy.
    """
    import nexusformat.nexus.tree as nx

    cal = nx.NXgroup()
    cal.nxclass = "NXcalibration"
    cal.attrs["description"] = "ramanchada2 " + component_dict.get("type", "calibration")
    cal.attrs["physical_quantity"] = (
        "relative intensity" if component_dict.get("type") == "YCalibrationComponent"
        else "wavenumber")
    cal.attrs["applied"] = bool(component_dict.get("enabled", True))
    cal.attrs["fit_formula_description"] = _fit_formula_description(component_dict)

    if curve is not None:
        original_axis, calibrated_axis = curve
        cal["original_axis"] = np.asarray(original_axis, dtype=float)
        cal["calibrated_axis"] = np.asarray(calibrated_axis, dtype=float)

    params = _component_parameters(component_dict)
    if params:
        pgroup = nx.NXgroup()
        pgroup.nxclass = "NXparameters"
        for key, value in params.items():
            if value is None:
                continue
            pgroup[key] = _h5_clean(value)
        cal["calibration_parameters"] = pgroup

    note = nx.NXnote()
    note["type"] = "application/json"
    note["data"] = json.dumps(_residual_component_fields(component_dict))
    cal["calibration_object"] = note

    anchors = component_dict.get("anchors")
    if anchors:
        agroup = nx.NXdata()
        agroup.attrs["description"] = "Matched calibrant peaks used to derive the model."
        arr = np.asarray(anchors, dtype=float)
        agroup["measured"] = arr[:, 0]
        agroup["reference"] = arr[:, 1]
        agroup["inlier"] = arr[:, 2].astype(bool)
        cal["anchors"] = agroup

    return cal


# Back-compat alias: the pre-existing name, now the richer calibration-workflow exporter.
# The old export_nexus was unreachable dead code (imported nowhere in this repo or its
# known callers), so no behavior is broken by widening its signature under this name.
export_nexus = export_nexus_calibration
