#!/usr/bin/env python3
"""The Neon x-calibration curve must honour the requested ``interpolator_method``.

Regression test for a bug where ``XCalibrationComponent.derive_model`` hardcoded
``interpolator_method="pchip"`` and silently ignored the configured value, so every
calibration (poly/polyinverse/...) was built as a PCHIP curve. The type is checked on
the freshly *built* model (not a loaded pickle, whose ``from_dict`` would normalise the type).
"""
import pytest

import ramanchada2.misc.constants as rc2const
from ramanchada2.protocols.calibration.calibration_model import CalibrationModel
from ramanchada2.protocols.calibration.interpolators import (
    CustomPChipInterpolator,
    CustomPolyInterpolator,
)
from ramanchada2.spectrum import from_test_spe

LASER_WL = 785


@pytest.fixture(scope="module")
def spe_neon():
    spe = from_test_spe(
        sample=["Neon"], provider=["ICV"], device=["BWtek"],
        OP=["100"], laser_wl=[str(LASER_WL)],
    )
    spe = spe.trim_axes(method="x-axis", boundaries=(100, max(spe.x)))
    return spe.subtract_baseline_rc1_snip(niter=40)


def _build_neon_model(spe_neon, interpolator_method):
    cm = CalibrationModel(LASER_WL)
    cm.nonmonotonic = "drop"
    cm.prominence_coeff = 3
    find_kw = {"wlen": 200, "width": 1,
               "prominence": spe_neon.y_noise_MAD() * 3}
    return cm.derive_model_curve(
        spe=spe_neon,
        ref=rc2const.NEON_WL[LASER_WL],
        spe_units="cm-1",
        ref_units="nm",
        find_kw=find_kw,
        fit_peaks_kw={},
        should_fit=False,
        name="Neon",
        match_method="cluster",
        interpolator_method=interpolator_method,
        extrapolate=True,
    )


@pytest.mark.parametrize("interpolator_method, expected_cls", [
    ("pchip", CustomPChipInterpolator),
    ("pchipinverse", CustomPChipInterpolator),
    ("poly", CustomPolyInterpolator),
    ("polyinverse", CustomPolyInterpolator),
    ("pchippolyinverse", CustomPolyInterpolator),  # deprecated alias of polyinverse
    # rbfinverse: thin-plate spline shape remapped through a monotone PCHIP (CWA/MATLAB
    # recipe), so the built model is a CustomPChipInterpolator.
    ("rbfinverse", CustomPChipInterpolator),
])
def test_interpolator_method_is_honoured(spe_neon, interpolator_method, expected_cls):
    model_ne = _build_neon_model(spe_neon, interpolator_method)
    built = type(model_ne.model)
    assert built is expected_cls, (
        f"interpolator_method={interpolator_method!r} built {built.__name__}, "
        f"expected {expected_cls.__name__}"
    )
