#!/usr/bin/env python3
"""Tests for export_nexus_calibration: the calibration workflow as a NeXus/HDF5 file.

Reuses the same real-data fixtures as test_calmodel_serialization.py (from_test_spe /
calibration_model_factory), so these exercise a genuine derived model, not a synthetic
stand-in.
"""

import json

import h5py
import numpy as np
import pytest

import ramanchada2.misc.constants as rc2const
from ramanchada2.protocols.calibration.calibration_model import CalibrationModel
from ramanchada2.protocols.calibration.serialization import export_nexus_calibration
from ramanchada2.protocols.calibration.ycalibration import (
    YCalibrationCertificate,
    YCalibrationComponent,
)
from ramanchada2.spectrum import from_test_spe

LASER_WL = 785
_SPE_KW = dict(provider=["ICV"], device=["BWtek"], OP=["100"], laser_wl=[str(LASER_WL)])


def _load(sample):
    return from_test_spe(sample=[sample], **_SPE_KW)


@pytest.fixture(scope="module")
def spe_neon():
    return _load("Neon").trim_axes(method="x-axis", boundaries=(100, 3500))


@pytest.fixture(scope="module")
def spe_sil():
    return _load("S0B").trim_axes(method="x-axis", boundaries=(520.45 - 50, 520.45 + 50))


@pytest.fixture(scope="module", params=["poly", "pchipinverse"])
def calmodel(request, spe_neon, spe_sil):
    spe_neon_bl = spe_neon.subtract_baseline_rc1_snip(niter=40)
    spe_sil_bl = spe_sil.subtract_baseline_rc1_snip(niter=40)
    return CalibrationModel.calibration_model_factory(
        LASER_WL,
        spe_neon_bl,
        spe_sil_bl,
        neon_wl=rc2const.NEON_WL[LASER_WL],
        find_kw={"wlen": 200, "width": 1},
        fit_peaks_kw={},
        should_fit=False,
        interpolator_method=request.param,
    )


@pytest.fixture(scope="module")
def ycal():
    cert = YCalibrationCertificate.load(wavelength=LASER_WL, key="NIST785_SRM2241")
    spe_srm = _load("NIST785_SRM2241")
    return YCalibrationComponent(LASER_WL, reference_spe_xcalibrated=spe_srm, certificate=cert)


@pytest.fixture(scope="module")
def spe_pst():
    return _load("PST").trim_axes(method="x-axis", boundaries=(100, 3500))


def _write(calmodel, tmp_path, **kw):
    path = str(tmp_path / "calibration.nxs")
    export_nexus_calibration(calmodel, path, spectral_range=(150.0, 3400.0), npoints=101, **kw)
    return path


def _entry_path(h5file):
    """The one NXentry in the file."""
    names = [k for k, v in h5file.items()
             if v.attrs.get("NX_class") in (b"NXentry", "NXentry")]
    assert len(names) == 1, f"expected exactly one NXentry, found {names}"
    return names[0]


def test_entry_is_nxraman(calmodel, tmp_path):
    """The entry declares the NXraman application definition."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = f[_entry_path(f)]
        assert entry["definition"][()].decode() == "NXraman"
        assert entry["experiment_type"][()].decode() == "Raman spectroscopy"
        assert entry["instrument"].attrs["NX_class"] == "NXinstrument"
        assert entry["sample"].attrs["NX_class"] == "NXsample"


def test_nexusformat_can_open(calmodel, tmp_path):
    """Structural sanity: the file is a valid NeXus tree, not just valid HDF5.

    nexusformat consumes the ``NX_class`` HDF5 attribute into its own ``.nxclass``
    property (it is not left visible via ``.attrs``), so a passing ``.nxclass`` check here
    is what proves nexusformat parsed our groups as real NeXus classes rather than opaque
    HDF5 groups.
    """
    nexusformat = pytest.importorskip("nexusformat.nexus")
    path = _write(calmodel, tmp_path)
    root = nexusformat.nxload(path)
    entry = next(e for e in root.entries.values() if e.nxclass == "NXentry")
    assert entry.instrument.nxclass == "NXinstrument"
    assert entry.instrument.calibration_x_0.nxclass == "NXcalibration"
    assert entry.calibration_curve.nxclass == "NXdata"


def test_calibration_object_json_roundtrips(calmodel, spe_pst, tmp_path):
    """The full model, reconstructed from the file alone, reproduces the calibrated axis."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        doc = json.loads(f[f"{entry}/calibration_model/data"][()])
    reconstructed = CalibrationModel.from_dict(doc["model"])

    orig = calmodel.apply_calibration_x(spe_pst, spe_units="cm-1").x
    rt = reconstructed.apply_calibration_x(spe_pst, spe_units="cm-1").x
    np.testing.assert_allclose(rt, orig, rtol=0, atol=1e-9)


def test_calibration_parameters_written(calmodel, tmp_path):
    """The reconstructable fit parameters are real numeric datasets, not just JSON text."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        group = f[f"{entry}/instrument/calibration_x_0/calibration_parameters"]
        kind = calmodel.components[0].to_dict()["model"]["type"]
        if kind == "CustomPolyInterpolator":
            assert "coef" in group and group["coef"].shape[0] >= 1
            assert int(group["degree"][()]) >= 1
            assert group["x_min"][()] < group["x_max"][()]
        elif kind == "CustomPChipInterpolator":
            assert group["knots_x"].shape == group["knots_y"].shape
            assert group["knots_x"].shape[0] > 1
        else:
            pytest.fail(f"unexpected interpolator kind {kind!r}")

        si_group = f[f"{entry}/instrument/calibration_x_1/calibration_parameters"]
        assert si_group["si_peak_nm"][()] > 0


def test_original_and_calibrated_axis_same_length(calmodel, tmp_path):
    """The sampled curve datasets satisfy the NXDL `ncal` symbol constraint (equal length)."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        original = f[f"{entry}/instrument/calibration_x_0/original_axis"]
        calibrated = f[f"{entry}/instrument/calibration_x_0/calibrated_axis"]
        assert original.shape == calibrated.shape
        assert original.shape[0] == 101  # npoints passed to _write


def test_anchors_not_in_calibrated_axis(calmodel, tmp_path):
    """Matched-peak anchors (a different length than the sampled curve) live in their own
    group, not mixed into original_axis/calibrated_axis."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        cal = f[f"{entry}/instrument/calibration_x_0"]
        assert "anchors" in cal
        anchors = cal["anchors"]
        assert set(anchors.keys()) >= {"measured", "reference", "inlier"}
        assert anchors["measured"].shape == anchors["reference"].shape


def test_reference_spectra_written(calmodel, spe_neon, spe_sil, tmp_path):
    """Regression guard: the calibrant x AND y arrays are actually persisted (the failure
    mode this guards against is ramanchada2.io.HSDS.write_nexus silently dropping the axis
    via require_group(..., data=x) — a group has no data= kwarg). Each calibrant is its own
    plottable NXdata group under the entry."""
    path = _write(calmodel, tmp_path, spe_neon=spe_neon, spe_neon_units="cm-1",
                  spe_silicon=spe_sil, spe_silicon_units="cm-1")
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        neon = f[f"{entry}/reference_neon"]
        assert neon.attrs["NX_class"] == "NXdata"
        assert neon.attrs["signal"] == "intensity"
        np.testing.assert_allclose(neon["raman_shift"][()], np.asarray(spe_neon.x))
        np.testing.assert_allclose(neon["intensity"][()], np.asarray(spe_neon.y))

        sil = f[f"{entry}/reference_silicon"]
        np.testing.assert_allclose(sil["raman_shift"][()], np.asarray(spe_sil.x))
        np.testing.assert_allclose(sil["intensity"][()], np.asarray(spe_sil.y))


def test_reference_spectra_omitted_when_not_given(calmodel, tmp_path):
    """No invented data: no reference-spectrum group exists if the caller didn't supply
    calibrant spectra."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        assert "reference_neon" not in f[entry]
        assert "reference_silicon" not in f[entry]


def test_ycal_component_written_when_given(calmodel, ycal, spe_pst, tmp_path):
    path = _write(calmodel, tmp_path, ycal_component=ycal)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        ycal_group = f[f"{entry}/instrument/calibration_y"]
        assert ycal_group.attrs["physical_quantity"] == "relative intensity"
        doc = json.loads(ycal_group["calibration_object/data"][()])
        reconstructed = YCalibrationComponent.from_dict(doc)
        spe = spe_pst.trim_axes(method="x-axis", boundaries=ycal.ref.raman_shift)
        np.testing.assert_allclose(
            reconstructed.process(spe).y, ycal.process(spe).y, rtol=0, atol=1e-9)


def test_ycal_plottable_curves_present(calmodel, ycal, tmp_path):
    """Regression: the y-calibration must be visually discoverable, not just present as
    fit parameters/JSON. The x side already had a plottable calibration_curve NXdata;
    the y side previously had none, which is why a user opening the file couldn't find
    the intensity calibration at all despite the model being fully written."""
    path = _write(calmodel, tmp_path, ycal_component=ycal)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        assert "calibration_curve_y" in f[entry], (
            "no plottable NXdata for the y (relative-intensity) calibration curve")
        y_curve = f[f"{entry}/calibration_curve_y"]
        assert y_curve.attrs["NX_class"] == "NXdata"
        assert y_curve.attrs["signal"] == "intensity_factor"
        assert y_curve["intensity_factor"].shape[0] > 1

        # the measured SRM response and the certificate's analytic response are both
        # written as their own NXdata groups, not just embedded numbers in the JSON
        for name in ("calibration_curve_y_measured", "calibration_curve_y_certificate"):
            assert name in f[entry], f"{name} not written"
            assert f[f"{entry}/{name}"].attrs["NX_class"] == "NXdata"


def test_ycal_component_omitted_when_not_given(calmodel, tmp_path):
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        assert "calibration_y" not in f[f"{entry}/instrument"]


def test_default_attr_points_at_calibration_curve(calmodel, tmp_path):
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        default_name = f[entry].attrs["default"]
        assert default_name in f[entry]
        assert f[entry][default_name].attrs["NX_class"] == "NXdata"


def test_instrument_metadata_written_and_nan_dropped(calmodel, tmp_path):
    """instrument_make/instrument_model become the NXfabrication (vendor, model) device
    identity, laser_wl the incident beam wavelength; other keys go to instrument/
    parameters. NaN values are dropped."""
    path = _write(calmodel, tmp_path, instrument={
        "instrument_make": "TestCo", "instrument_model": "ModelX",
        "grating": float("nan"), "laser_wl": 785,
    })
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        device = f[f"{entry}/instrument/device_information"]
        assert device["vendor"][()].decode() == "TestCo"
        assert device["model"][()].decode() == "ModelX"
        # grating was NaN -> dropped; must not appear anywhere as a written value
        assert f[f"{entry}/instrument/beam_incident/wavelength"][()] == 785
        assert f[f"{entry}/instrument/beam_incident/wavelength"].attrs["units"] == "nm"
        assert "parameters" not in f[f"{entry}/instrument"]


def test_h5pyd_style_module_is_used(calmodel, tmp_path):
    """h5module is honoured (h5pyd exposes the h5py File/require_group/attrs API)."""
    import h5py
    calls = []

    class _Spy:
        def __getattr__(self, name):
            return getattr(h5py, name)

        def File(self, *a, **kw):
            calls.append(a)
            return h5py.File(*a, **kw)

    path = str(tmp_path / "spy.nxs")
    export_nexus_calibration(calmodel, path, npoints=11, h5module=_Spy())
    assert calls and calls[0][0] == path
