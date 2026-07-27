"""Relative-intensity calibration: fit the measured reference with the
certificate's own functional form.

The measured SRM/LED reference is fitted with a polynomial of the certificate's
order (polynomial certificates) or the certificate's log-Gaussian (the SRM glass
certificates), seeded by the certificate's own parameters. The analytic fit
denoises the reference without smoothing. Driven with synthetic references of
the certificate's shape plus noise, so the outcome is deterministic and fast.
"""

import numpy as np
import pytest

from ramanchada2.protocols.calibration.interpolators import (
    CustomPChipInterpolator,
    ParametricModel,
)
from ramanchada2.protocols.calibration.ycalibration import (
    CertificatesDict,
    YCalibrationComponent,
)
from ramanchada2.spectrum import Spectrum

POLY_CERT = ("785", "NIST785_SRM2241")  # A0..A5 polynomial
LOG_GAUSS_CERT = ("532", "NIST532_SRM2242a")  # H*exp(...)+m*x+b


def _noisy_reference(cert, scale=1000.0, noise=0.03, seed=0):
    """A measured reference of the certificate's shape, scaled and noised."""
    lo, hi = cert.raman_shift
    x = np.linspace(lo, hi, 500)
    truth = np.asarray(cert.Y(x), dtype=float)
    rng = np.random.default_rng(seed)
    y = truth * scale + rng.normal(0, noise * np.max(truth) * scale, x.size)
    return x, y, truth


def test_certificate_polynomial_order():
    poly = CertificatesDict.load(*POLY_CERT)
    assert poly.polynomial_order == 5
    log_g = CertificatesDict.load(*LOG_GAUSS_CERT)
    assert log_g.polynomial_order is None
    # helpers parse the certificate's own parameters
    assert log_g.param_names == ["H", "w", "ro", "x0", "m", "b"]
    assert len(log_g.param_values) == 6


@pytest.mark.parametrize("wl,key", [POLY_CERT, LOG_GAUSS_CERT])
def test_reference_fitted_with_certificate_form(wl, key):
    cert = CertificatesDict.load(wl, key)
    x, y, truth = _noisy_reference(cert)
    ycal = YCalibrationComponent(int(wl), Spectrum(x=x, y=y), cert)

    # the measured reference is now an analytic model of the certificate's form
    assert isinstance(ycal.model, ParametricModel)

    fit = np.asarray(ycal.model(x), dtype=float)
    # the fit is far smoother than the raw measured reference (denoised)...
    raw_p2p = np.std(np.diff(y / np.max(np.abs(y))))
    fit_p2p = np.std(np.diff(fit))
    assert fit_p2p < raw_p2p / 5
    # ...and recovers the noise-free certificate shape (normalized)
    assert np.max(np.abs(fit - truth / np.max(truth))) < 0.1
    assert np.all(np.isfinite(fit))


@pytest.mark.parametrize("wl,key", [POLY_CERT, LOG_GAUSS_CERT])
def test_fitted_model_round_trips(wl, key):
    cert = CertificatesDict.load(wl, key)
    x, y, _ = _noisy_reference(cert)
    ycal = YCalibrationComponent(int(wl), Spectrum(x=x, y=y), cert)
    reloaded = YCalibrationComponent.from_dict(ycal.to_dict())
    assert isinstance(reloaded.model, ParametricModel)
    np.testing.assert_allclose(ycal.model(x), reloaded.model(x))


def test_process_is_finite_over_the_certified_range():
    wl, key = LOG_GAUSS_CERT
    cert = CertificatesDict.load(wl, key)
    x, y, _ = _noisy_reference(cert)
    ycal = YCalibrationComponent(int(wl), Spectrum(x=x, y=y), cert)
    out = ycal.process(Spectrum(x=x, y=np.ones_like(x)))
    assert np.all(np.isfinite(out.y))
    assert (out.y != 0).mean() > 0.9  # applied across the certified range


def test_pchip_method_and_fallback():
    wl, key = POLY_CERT
    cert = CertificatesDict.load(wl, key)
    x, y, _ = _noisy_reference(cert)
    # explicit legacy path
    legacy = YCalibrationComponent(
        int(wl), Spectrum(x=x, y=y), cert, model_method="pchip"
    )
    assert isinstance(legacy.model, CustomPChipInterpolator)
    # too few points to fit -> silent fallback to PCHIP, no crash
    few = YCalibrationComponent(
        int(wl), Spectrum(x=x[:4], y=y[:4]), cert, model_method="certificate"
    )
    assert isinstance(few.model, CustomPChipInterpolator)


def test_fit_order_override_forces_polynomial():
    wl, key = LOG_GAUSS_CERT  # a non-polynomial certificate...
    cert = CertificatesDict.load(wl, key)
    x, y, _ = _noisy_reference(cert)
    # ...forced to a degree-4 polynomial fit
    ycal = YCalibrationComponent(int(wl), Spectrum(x=x, y=y), cert, fit_order=4)
    assert isinstance(ycal.model, ParametricModel)
    assert len(ycal.model.coef) == 5  # A0..A4
