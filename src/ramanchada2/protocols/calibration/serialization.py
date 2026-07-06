"""CWA 18133:2024 §8 portable calibration files.

§8 asks for a calibration file with metadata, date, the calibration curve as points
(uncalibrated shift -> calibrated shift), the silicon peak position and (optionally) the
calibrated laser wavelength; the curve points are meant to regenerate a spline. These
exporters write that as ``<base>.csv`` (the curve) + ``<base>.json`` (metadata AND the full
model in the portable :meth:`CalibrationModel.to_dict` form, so the exact model — not just
the sampled curve — can be reconstructed by ramanchada2 or any JSON-capable reader).

``export_nexus`` optionally embeds the same content in a NeXus/HDF5 file.
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
        "laser_wl_nominal_nm": calmodel.laser_wl,
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
        "laser_wl_nominal_nm": ycal_component.laser_wl,
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


def export_nexus(calmodel, filename, spectral_range=(100, 3500), npoints=200, metadata=None):
    """Optional NeXus/HDF5 export: curve as NXdata, portable model JSON as NXnote."""
    import h5py

    grid, calibrated = _calibration_curve(calmodel, spectral_range, npoints)
    with h5py.File(filename, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "calibration_curve"

        process = entry.create_group("process")
        process.attrs["NX_class"] = "NXprocess"
        process.create_dataset("program", data="ramanchada2")
        process.create_dataset("date", data=datetime.datetime.now().isoformat(timespec="seconds"))

        data = entry.create_group("calibration_curve")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "calibrated_cm1"
        data.attrs["axes"] = ["uncalibrated_cm1"]
        data.create_dataset("uncalibrated_cm1", data=grid)
        data.create_dataset("calibrated_cm1", data=calibrated)

        note = entry.create_group("calibration_model")
        note.attrs["NX_class"] = "NXnote"
        note.create_dataset("type", data="application/json")
        doc = {"metadata": metadata or {}, **_laser_zero_info(calmodel),
               "model": calmodel.to_dict()}
        note.create_dataset("data", data=json.dumps(doc))
    return filename
