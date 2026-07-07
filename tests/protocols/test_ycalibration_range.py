"""Regression tests for YCalibrationComponent range handling.

YCalibrationCertificate.raman_shift is Optional[Tuple[int, int]], but
YCalibrationComponent used to call trim_axes(boundaries=self.ref.raman_shift)
unconditionally in __init__/derive_model (crashes on None), and process()
evaluated the reference model on the full, untrimmed spectrum x-axis, so
points outside the certified range were assigned CustomPChipInterpolator's
unit-slope extrapolation (meant for x-calibration curves) as if it were a
real measured intensity factor.
"""
import numpy as np

from ramanchada2.protocols.calibration.ycalibration import (
    YCalibrationCertificate, YCalibrationComponent)
from ramanchada2.spectrum import from_test_spe

LASER_WL = 785
_SPE_KW = dict(provider=["ICV"], device=["BWtek"], OP=["100"], laser_wl=[str(LASER_WL)])


def _load(sample):
    return from_test_spe(sample=[sample], **_SPE_KW)


def _certificate(raman_shift=None):
    return YCalibrationCertificate(
        id="NIST785_SRM2241",
        wavelength=LASER_WL,
        params="A0 = 9.71937e-02, A1 = 2.28325e-04",
        equation="A0 + A1 * x",
        raman_shift=raman_shift,
    )


def test_no_certified_range_does_not_crash():
    """Certificates without a raman_shift must not crash construction."""
    spe_srm = _load("NIST785_SRM2241")
    cert = _certificate(raman_shift=None)
    ycal = YCalibrationComponent(LASER_WL, reference_spe_xcalibrated=spe_srm, certificate=cert)
    assert ycal.model is not None


def test_process_masks_outside_certified_range():
    """Points outside ref.raman_shift must not receive an extrapolated 'intensity'."""
    spe_srm = _load("NIST785_SRM2241")
    lo, hi = 200, 3000
    cert = _certificate(raman_shift=(lo, hi))
    ycal = YCalibrationComponent(LASER_WL, reference_spe_xcalibrated=spe_srm, certificate=cert)

    spe_pst = _load("PST")
    result = ycal.process(spe_pst)

    outside = (spe_pst.x < lo) | (spe_pst.x > hi)
    assert outside.any(), "test spectrum should extend outside the certified range"
    assert np.all(result.y[outside] == 0), \
        "points outside the certified range must be masked, not extrapolated"
