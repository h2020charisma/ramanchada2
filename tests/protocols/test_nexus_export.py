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


def test_build_measurement_papp_entry_is_discoverable(tmp_path):
    """Regression: pyambit's to_nexus() writes the entry under a leading-slash path
    derived from (provider, papp.uuid); it's reachable via nx_root.entries (the pattern
    pyambit's own test suite uses), not .keys(). Also: spe2ambit's per-effect group name
    comes from its `sample` argument (routed to spe2effect's nx_name) -- passing the same
    generic label for every spectrum makes every effect group indistinguishable except by
    position, so _build_measurement_papp must pass each spectrum's own nx_name there."""
    pytest.importorskip("pyambit")
    from ramanchada2.protocols.calibration.serialization import _build_measurement_papp
    import nexusformat.nexus.tree as nx

    spe_neon = _load("Neon").trim_axes(method="x-axis", boundaries=(100, 3500))
    papp = _build_measurement_papp(
        [(spe_neon.x, spe_neon.y, "RAW_DATA", "reference_neon", "cm-1")],
        meta=None, instrument=("TestCo", "ModelX"), wavelength=LASER_WL)
    assert papp is not None

    nx_root = nx.NXroot()
    papp.to_nexus(nx_root)
    entry = next(iter(nx_root.entries.values()))
    assert "instrument" in entry
    path = str(tmp_path / "papp_only.nxs")
    nx_root.save(path, mode="w")

    import h5py
    with h5py.File(path, "r") as f:
        groups = []
        f.visititems(lambda name, obj: groups.append(name))
    assert any("reference_neon" in g for g in groups), (
        f"reference_neon effect not found anywhere in the written tree: {groups}")


def _entry_path(h5file):
    """The one real NXentry in the file. pyambit's to_nexus writes it at a leading-slash
    path derived from (provider, papp.uuid), not a fixed "/entry" -- discover it instead
    of assuming the name (see test_build_measurement_papp_entry_is_discoverable)."""
    names = [k for k, v in h5file.items()
            if v.attrs.get("NX_class") in (b"NXentry", "NXentry")]
    assert len(names) == 1, f"expected exactly one NXentry, found {names}"
    return names[0]


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
    entry = next(iter(root.entries.values()))
    assert entry.nxclass == "NXentry"
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
    via require_group(..., data=x) — a group has no data= kwarg). Written through pyambit's
    spe2ambit/process_pa, so the effect lands under a RAW_DATA group, named by nx_name."""
    path = _write(calmodel, tmp_path, spe_neon=spe_neon, spe_neon_units="cm-1",
                  spe_silicon=spe_sil, spe_silicon_units="cm-1")
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        neon = f[f"{entry}/RAW_DATA/reference_neon_1"]
        np.testing.assert_allclose(neon["raman_shift"][()], np.asarray(spe_neon.x))
        np.testing.assert_allclose(neon["y"][()], np.asarray(spe_neon.y))

        sil = f[f"{entry}/RAW_DATA/reference_silicon_2"]
        np.testing.assert_allclose(sil["raman_shift"][()], np.asarray(spe_sil.x))
        np.testing.assert_allclose(sil["y"][()], np.asarray(spe_sil.y))


def test_reference_spectra_omitted_when_not_given(calmodel, tmp_path):
    """No invented data: no RAW_DATA reference-spectrum effect exists if the caller didn't
    supply calibrant spectra (the x-calibration curve itself still gets its own
    non-RAW_DATA group, so RAW_DATA is absent entirely rather than merely smaller)."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        assert "RAW_DATA" not in f[entry]


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
        # written as their own RAW_DATA-adjacent effects (via spe2ambit), not just
        # embedded numbers in calibration_object's JSON
        raw_names = list(f[f"{entry}/Y_CALIBRATION_MEASURED"].keys()) \
            if "Y_CALIBRATION_MEASURED" in f[entry] else []
        cert_names = list(f[f"{entry}/Y_CALIBRATION_CERTIFICATE"].keys()) \
            if "Y_CALIBRATION_CERTIFICATE" in f[entry] else []
        assert raw_names, "measured SRM response effect not written"
        assert cert_names, "certificate response effect not written"


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
    """instrument_make/instrument_model become pyambit's typed (vendor, model) device
    identity; other instrument keys (grating) route through configure_papp's backward-
    compat table into typed fields or the generic parameters bucket. NaN values must not
    reach either -- they're dropped before crossing into pyambit's meta dict."""
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
        params = f[f"{entry}/parameters"] if "parameters" in f[entry] else {}
        assert "grating" not in params
