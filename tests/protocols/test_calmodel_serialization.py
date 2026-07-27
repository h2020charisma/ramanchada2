#!/usr/bin/env python3
"""Round-trip tests for the portable (JSON+CSV, CWA 18133 §8) calibration files."""

import json

import numpy as np
import pytest

import ramanchada2.misc.constants as rc2const
from ramanchada2.protocols.calibration.calibration_model import CalibrationModel
from ramanchada2.protocols.calibration.serialization import export_cwa_x, export_cwa_y
from ramanchada2.protocols.calibration.ycalibration import (
    YCalibrationCertificate,
    YCalibrationComponent,
)
from ramanchada2.spectrum import from_test_spe

LASER_WL = 785
_SPE_KW = dict(provider=["ICV"], device=["BWtek"], OP=["100"], laser_wl=[str(LASER_WL)])


def _load(sample):
    return from_test_spe(sample=[sample], **_SPE_KW)


@pytest.fixture(scope="module", params=["poly", "pchipinverse", "pchippolyinverse"])
def calmodel(request):
    spe_neon = _load("Neon").trim_axes(method="x-axis", boundaries=(100, 3500))
    spe_sil = _load("S0B").trim_axes(method="x-axis", boundaries=(520.45 - 50, 520.45 + 50))
    spe_neon = spe_neon.subtract_baseline_rc1_snip(niter=40)
    spe_sil = spe_sil.subtract_baseline_rc1_snip(niter=40)
    return CalibrationModel.calibration_model_factory(
        LASER_WL,
        spe_neon,
        spe_sil,
        neon_wl=rc2const.NEON_WL[LASER_WL],
        find_kw={"wlen": 200, "width": 1},
        fit_peaks_kw={},
        should_fit=False,
        interpolator_method=request.param,
    )


@pytest.fixture(scope="module")
def spe_pst():
    return _load("PST").trim_axes(method="x-axis", boundaries=(100, 3500))


def test_json_roundtrip_x(calmodel, spe_pst, tmp_path):
    path = str(tmp_path / "calmodel.json")
    calmodel.save(path)
    loaded = CalibrationModel.from_file(path)

    assert loaded.laser_wl == calmodel.laser_wl
    assert len(loaded.components) == len(calmodel.components)

    orig = calmodel.apply_calibration_x(spe_pst, spe_units="cm-1").x
    rt = loaded.apply_calibration_x(spe_pst, spe_units="cm-1").x
    np.testing.assert_allclose(rt, orig, rtol=0, atol=1e-9)


def test_json_matches_pickle(calmodel, spe_pst, tmp_path):
    jsn = str(tmp_path / "calmodel.json")
    pkl = str(tmp_path / "calmodel.pkl")
    calmodel.save(jsn)
    calmodel.save(pkl)
    from_json = CalibrationModel.from_file(jsn).apply_calibration_x(spe_pst, spe_units="cm-1").x
    from_pkl = CalibrationModel.from_file(pkl).apply_calibration_x(spe_pst, spe_units="cm-1").x
    np.testing.assert_allclose(from_json, from_pkl, rtol=0, atol=1e-9)


def test_json_is_portable(calmodel, tmp_path):
    """The file must be plain JSON with the documented structure (no pickles inside)."""
    path = str(tmp_path / "calmodel.json")
    calmodel.save(path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["format"] == "ramanchada2-calmodel"
    assert doc["laser_wl"] == LASER_WL
    types = [c["type"] for c in doc["components"]]
    assert types == ["XCalibrationComponent", "LazerZeroingComponent"]
    xcomp = doc["components"][0]
    assert "type" in xcomp["model"]
    # anchors provenance survives
    assert xcomp.get("anchors"), "matched anchors should be stored for provenance"
    # laser zeroing model is a plain float (Si peak position, nm)
    assert isinstance(doc["components"][1]["model"], float)


def test_export_cwa_x(calmodel, tmp_path):
    lo, hi = 150.0, 3400.0
    csv_path, json_path = export_cwa_x(
        calmodel, str(tmp_path / "cwa_x"), spectral_range=(lo, hi), npoints=101,
        metadata={"key": "TEST", "optical_path": "OP1"},
    )
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    assert rows.dtype.names == ("uncalibrated_cm1", "calibrated_cm1")
    assert rows["uncalibrated_cm1"][0] == pytest.approx(lo)
    assert rows["uncalibrated_cm1"][-1] == pytest.approx(hi)
    cal = rows["calibrated_cm1"]
    cal = cal[np.isfinite(cal)]
    assert np.all(np.diff(cal) > 0), "calibrated axis must stay monotonic"

    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["format"] == "CWA18133-x-calibration"
    assert doc["si_peak_nm"] > 0
    # calibrated laser wavelength should be near the nominal one
    assert doc["calibrated_laser_wl_nm"] == pytest.approx(LASER_WL, abs=2.0)
    assert doc["metadata"]["key"] == "TEST"
    # full model embedded: reconstructable from the CWA json alone
    model = CalibrationModel.from_dict(doc["model"])
    assert len(model.components) == 2


@pytest.fixture(scope="module")
def ycal():
    cert = YCalibrationCertificate.load(wavelength=LASER_WL, key="NIST785_SRM2241")
    spe_srm = _load("NIST785_SRM2241")
    return YCalibrationComponent(LASER_WL, reference_spe_xcalibrated=spe_srm, certificate=cert)


def test_json_roundtrip_y(ycal, spe_pst, tmp_path):
    d = ycal.to_dict()
    d = json.loads(json.dumps(d))  # force through JSON to catch non-serializable leftovers
    loaded = YCalibrationComponent.from_dict(d)
    spe = spe_pst.trim_axes(method="x-axis", boundaries=ycal.ref.raman_shift)
    orig = ycal.process(spe).y
    rt = loaded.process(spe).y
    np.testing.assert_allclose(rt, orig, rtol=0, atol=1e-9)


def test_export_cwa_y(ycal, tmp_path):
    csv_path, json_path = export_cwa_y(
        ycal, str(tmp_path / "cwa_y"), metadata={"key": "TEST"},
        x_calibration_ref="calmodel_785_OP1.json",
    )
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    assert rows.dtype.names == ("calibrated_cm1", "intensity_factor")
    assert np.any(rows["intensity_factor"] > 0)
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["format"] == "CWA18133-y-calibration"
    assert doc["certificate"]["id"] == "NIST785_SRM2241"
    assert doc["x_calibration_ref"] == "calmodel_785_OP1.json"
