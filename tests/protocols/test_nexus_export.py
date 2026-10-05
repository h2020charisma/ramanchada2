#!/usr/bin/env python3
"""Tests for export_nexus_calibration: the calibration workflow as a NeXus/HDF5 file.

Reuses the same real-data fixtures as test_calmodel_serialization.py (from_test_spe /
calibration_model_factory), so these exercise a genuine derived model, not a synthetic
stand-in.
"""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

import ramanchada2.misc.constants as rc2const
from ramanchada2.protocols.calibration.calibration_model import CalibrationModel
from ramanchada2.protocols.calibration.serialization import export_nexus_calibration
from ramanchada2.protocols.calibration.ycalibration import (
    YCalibrationCertificate,
    YCalibrationComponent,
)
from ramanchada2.spectrum import Spectrum, from_test_spe

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
    """The y-calibration is discoverable as plottable NXdata, with truthful names: the
    plottable ``calibration_curve_y`` is the intensity FACTOR (certificate / measured), the
    two responses it is built from are written under their own, differently named signals."""
    path = _write(calmodel, tmp_path, ycal_component=ycal)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        signals = {"calibration_curve_y": "intensity_factor",
                   "calibration_curve_y_measured": "measured_response",
                   "calibration_curve_y_certificate": "certificate_response"}
        for name, signal in signals.items():
            assert name in f[entry], f"{name} not written"
            group = f[f"{entry}/{name}"]
            assert group.attrs["NX_class"] == "NXdata"
            assert group.attrs["signal"] == signal
            assert group[signal].shape == (101,)
            assert group["calibrated_cm1"].attrs["units"] == "1/cm"


def test_ycal_curve_values(calmodel, ycal, tmp_path):
    """The y curves hold the right VALUES: factor == certificate / measured (what
    YCalibrationComponent.process applies), not the raw measured response mislabeled."""
    path = _write(calmodel, tmp_path, ycal_component=ycal)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        x = f[f"{entry}/calibration_curve_y/calibrated_cm1"][()]
        factor = f[f"{entry}/calibration_curve_y/intensity_factor"][()]
        measured = f[f"{entry}/calibration_curve_y_measured/measured_response"][()]
        certificate = f[f"{entry}/calibration_curve_y_certificate/certificate_response"][()]
        np.testing.assert_array_equal(
            x, f[f"{entry}/calibration_curve_y_measured/calibrated_cm1"][()])

    lo, hi = ycal.ref.raman_shift
    np.testing.assert_allclose(x, np.linspace(lo, hi, 101))
    np.testing.assert_allclose(measured, ycal.model(x), rtol=1e-12)
    np.testing.assert_allclose(certificate, ycal.ref.Y(x), rtol=1e-12)

    used = factor != 0
    assert used.sum() > 50, "factor is (almost) entirely masked"
    np.testing.assert_allclose(factor[used], certificate[used] / measured[used], rtol=1e-9)
    # not the mislabeled raw measured response
    assert not np.allclose(factor, measured)
    # it is exactly what process() multiplies a spectrum by
    applied = ycal.process(Spectrum(x=x.copy(), y=np.full_like(x, 3.0))).y
    np.testing.assert_allclose(applied, 3.0 * factor, rtol=1e-12)


def test_ycal_curve_failure_is_not_silent(calmodel, ycal, tmp_path, monkeypatch, caplog):
    """If the y curves cannot be evaluated, a warning is logged and the calibration_y group
    records it; unexpected error types propagate instead of being swallowed."""
    def _boom(*a, **kw):
        raise ValueError("cannot evaluate")

    monkeypatch.setattr(ycal, "process", _boom)
    with caplog.at_level("WARNING", logger="ramanchada2.protocols.calibration.serialization"):
        path = _write(calmodel, tmp_path, ycal_component=ycal)
    assert any("y calibration curves not written" in r.getMessage() for r in caplog.records)
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        group = f[f"{entry}/instrument/calibration_y"]
        assert "cannot evaluate" in group.attrs["curves_error"]
        for name in ("calibration_curve_y", "calibration_curve_y_measured",
                     "calibration_curve_y_certificate"):
            assert name not in f[entry]

    def _bug(*a, **kw):
        raise TypeError("programming error")

    monkeypatch.setattr(ycal, "process", _bug)
    with pytest.raises(TypeError):
        _write(calmodel, tmp_path, ycal_component=ycal)


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


def test_instrument_metadata_coercion(calmodel, tmp_path):
    """numpy/pandas values are written as sensible HDF5 values; missing values of any float
    width / pd.NA / pd.NaT are dropped; '/' in keys does not create nested groups."""
    path = _write(calmodel, tmp_path, instrument={
        "instrument_make": np.str_("NumpyCo"),
        "instrument_model": "M",
        "slit_size": np.int64(50),
        "grating": np.float32(1200.5),
        "hole": np.float64(0.25),
        "nan32": np.float32("nan"),
        "nan16": np.float16("nan"),
        "nan64": np.float64("nan"),
        "na": pd.NA,
        "nat": pd.NaT,
        "none": None,
        "date": pd.Timestamp("2024-05-06 07:08:09"),
        "tags": ["a", "b"],
        "np_tags": np.array(["x", "yy"]),
        "ranges": [1, 2, 3],
        "mixed": [1, "a"],
        "flag": np.bool_(True),
        "a/b": "slashed",
    })
    with h5py.File(path, "r") as f:
        entry = _entry_path(f)
        assert entry == "entry"
        assert f[f"{entry}/instrument/device_information/vendor"][()].decode() == "NumpyCo"
        params = f[f"{entry}/instrument/parameters"]
        assert set(params.keys()) == {"slit_size", "grating", "hole", "date", "tags", "np_tags",
                                      "ranges", "mixed", "flag", "a_b"}
        assert params["slit_size"][()] == 50
        assert params["grating"][()] == pytest.approx(1200.5)
        assert params["hole"][()] == 0.25
        assert params["date"][()].decode() == "2024-05-06T07:08:09"
        assert [v.decode() for v in params["tags"][()]] == ["a", "b"]
        assert [v.decode() for v in params["np_tags"][()]] == ["x", "yy"]
        np.testing.assert_array_equal(params["ranges"][()], [1, 2, 3])
        assert json.loads(params["mixed"][()]) == [1, "a"]
        assert bool(params["flag"][()]) is True
        assert params["a_b"][()].decode() == "slashed"


def test_metadata_with_numpy_and_timestamps_serialises(calmodel, tmp_path):
    """The JSON model note must not choke on numpy / pandas values in ``metadata``."""
    path = _write(calmodel, tmp_path, metadata={
        "n": np.int64(3), "x": np.float32(1.5), "when": pd.Timestamp("2024-01-02"),
        "arr": np.arange(3), "missing": pd.NA})
    with h5py.File(path, "r") as f:
        doc = json.loads(f["entry/calibration_model/data"][()])
    assert doc["metadata"] == {"n": 3, "x": 1.5, "when": "2024-01-02T00:00:00",
                               "arr": [0, 1, 2], "missing": None}


def test_beam_wavelength_falls_back_to_calmodel(calmodel, tmp_path):
    """No ``wavelength`` and no instrument laser_wl: use calmodel.laser_wl. An explicit
    argument wins; a non-numeric instrument laser_wl is kept as a parameter and ignored."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        assert f["entry/instrument/beam_incident/wavelength"][()] == LASER_WL
    path = _write(calmodel, tmp_path, wavelength=633, instrument={"laser_wl": 785})
    with h5py.File(path, "r") as f:
        assert f["entry/instrument/beam_incident/wavelength"][()] == 633
    path = _write(calmodel, tmp_path, instrument={"laser_wl": "n/a"})
    with h5py.File(path, "r") as f:
        assert f["entry/instrument/beam_incident/wavelength"][()] == LASER_WL
        assert f["entry/instrument/parameters/laser_wl"][()].decode() == "n/a"


def test_units_are_nexus_spelling_everywhere(calmodel, spe_neon, tmp_path):
    """One NeXus-valid spelling ("1/cm") for wavenumber axes, in every NXdata group."""
    path = _write(calmodel, tmp_path, spe_neon=spe_neon, spe_neon_units="cm-1")
    with h5py.File(path, "r") as f:
        curve = f["entry/calibration_curve"]
        assert curve["uncalibrated_cm1"].attrs["units"] == "1/cm"
        assert curve["calibrated_cm1"].attrs["units"] == "1/cm"
        assert f["entry/reference_neon/raman_shift"].attrs["units"] == "1/cm"
        seen = set()
        f.visititems(lambda n, o: seen.add(o.attrs["units"])
                     if isinstance(o, h5py.Dataset) and "units" in o.attrs else None)
        assert seen == {"1/cm", "nm"}


def test_reference_spectrum_in_nm_keeps_nm_units(calmodel, spe_neon, tmp_path):
    path = _write(calmodel, tmp_path, spe_neon=spe_neon, spe_neon_units="nm")
    with h5py.File(path, "r") as f:
        assert f["entry/reference_neon/wavelength"].attrs["units"] == "nm"


def test_stage_curves_belong_to_their_component(calmodel, tmp_path):
    """calibration_x_<i>/original_axis -> calibrated_axis is that component's OWN stage:
    Ne maps cm-1 -> nm, Si zeroing maps nm -> cm-1, and chaining them gives the end-to-end
    calibration_curve."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        curve_x = f["entry/calibration_curve/uncalibrated_cm1"][()]
        curve_y = f["entry/calibration_curve/calibrated_cm1"][()]
        ne = f["entry/instrument/calibration_x_0"]
        si = f["entry/instrument/calibration_x_1"]
        assert ne["original_axis"].attrs["units"] == "1/cm"
        assert ne["calibrated_axis"].attrs["units"] == "nm"
        assert si["original_axis"].attrs["units"] == "nm"
        assert si["calibrated_axis"].attrs["units"] == "1/cm"
        np.testing.assert_allclose(ne["original_axis"][()], curve_x)
        np.testing.assert_allclose(si["original_axis"][()], ne["calibrated_axis"][()])
        np.testing.assert_allclose(si["calibrated_axis"][()], curve_y)

        # independent recomputation of the Ne stage alone
        ne_comp = calmodel.components[0]
        expected_nm = ne_comp.process(Spectrum(x=curve_x.copy(), y=np.ones_like(curve_x)),
                                      "cm-1").x
        np.testing.assert_allclose(ne["calibrated_axis"][()], expected_nm)
        # nm values are physically sensible for a 785 nm laser (Stokes shifts 150-3400 cm-1)
        assert 785 < ne["calibrated_axis"][()].min() < ne["calibrated_axis"][()].max() < 1100


def test_h5pyd_style_module_is_used(calmodel, ycal, tmp_path, monkeypatch):
    """h5module is honoured for EVERYTHING (File, string dtypes): a stray direct ``h5py``
    use inside the exporter would hit the patched h5py entry points and raise."""
    real = {name: getattr(h5py, name) for name in ("File", "string_dtype", "special_dtype")}
    calls = []

    class _Spy:
        def __getattr__(self, name):
            return real[name]

        def File(self, *a, **kw):
            calls.append(("File", a))
            return real["File"](*a, **kw)

        def string_dtype(self, *a, **kw):
            calls.append(("string_dtype", a))
            return real["string_dtype"](*a, **kw)

    def _forbidden(name):
        def _raise(*a, **kw):
            raise AssertionError(f"exporter used h5py.{name} directly instead of h5module")
        return _raise

    for name in real:
        monkeypatch.setattr(h5py, name, _forbidden(name))

    path = str(tmp_path / "spy.nxs")
    # ycal component + string-list instrument metadata exercise the string_dtype path
    export_nexus_calibration(calmodel, path, npoints=11, h5module=_Spy(),
                             ycal_component=ycal, instrument={"tags": ["a", "b"]})
    assert ("File", (path, "w")) in calls
    assert any(c[0] == "string_dtype" for c in calls)
    monkeypatch.undo()
    with h5py.File(path, "r") as f:
        assert f["entry/definition"][()].decode() == "NXraman"


def test_failed_export_keeps_existing_file_and_leaves_no_temp(calmodel, tmp_path, monkeypatch):
    """A failure while writing must neither truncate an existing file nor leave a .tmp."""
    import ramanchada2.protocols.calibration.serialization as ser

    path = tmp_path / "calibration.nxs"
    path.write_bytes(b"previous content")

    def _boom(_calmodel):
        raise RuntimeError("boom")

    monkeypatch.setattr(ser, "_laser_zero_info", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        export_nexus_calibration(calmodel, str(path), npoints=11)

    assert path.read_bytes() == b"previous content"
    assert not (tmp_path / "calibration.nxs.tmp").exists()


def test_successful_export_leaves_no_temp(calmodel, tmp_path):
    path = _write(calmodel, tmp_path)
    assert h5py.is_hdf5(path)
    assert not (tmp_path / "calibration.nxs.tmp").exists()


def test_vendor_model_options_override_instrument_dict(calmodel, tmp_path):
    path = _write(calmodel, tmp_path, vendor="OptCo", model="OptModel",
                  instrument={"instrument_make": "DictCo", "instrument_model": "DictModel"})
    with h5py.File(path, "r") as f:
        device = f[f"{_entry_path(f)}/instrument/device_information"]
        assert device["vendor"][()].decode() == "OptCo"
        assert device["model"][()].decode() == "OptModel"


def test_vendor_model_not_invented_when_unset(calmodel, tmp_path):
    """No hard-coded "unknown": without vendor/model the device group is simply absent."""
    path = _write(calmodel, tmp_path)
    with h5py.File(path, "r") as f:
        assert "device_information" not in f[f"{_entry_path(f)}/instrument"]


def test_only_vendor_given_writes_only_vendor(calmodel, tmp_path):
    path = _write(calmodel, tmp_path, vendor="OnlyCo")
    with h5py.File(path, "r") as f:
        device = f[f"{_entry_path(f)}/instrument/device_information"]
        assert set(device.keys()) == {"vendor"}


def test_laser_wl_not_duplicated_as_parameter(calmodel, tmp_path):
    """An instrument laser_wl that merely repeats the (explicit) beam wavelength is not
    also written under instrument/parameters; a conflicting one is kept."""
    path = _write(calmodel, tmp_path, wavelength=785, instrument={"laser_wl": 785, "grating": 600})
    with h5py.File(path, "r") as f:
        assert f["entry/instrument/beam_incident/wavelength"][()] == 785
        assert "laser_wl" not in f["entry/instrument/parameters"]
        assert "grating" in f["entry/instrument/parameters"]
    path = _write(calmodel, tmp_path, wavelength=633, instrument={"laser_wl": 785})
    with h5py.File(path, "r") as f:
        assert f["entry/instrument/parameters/laser_wl"][()] == 785
