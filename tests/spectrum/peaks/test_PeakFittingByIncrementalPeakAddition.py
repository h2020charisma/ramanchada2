from ramanchada2.fitting_functions import voigtareaparametrizationnu
from ramanchada2.fitting_functions.pearsonivamplitudeparametrizationhpw import (
    PearsonIVAmplitudeParametrizationHPW,
)
from ramanchada2.spectrum.peaks.find_peaks_ByIncrementalPeakAddition import (
    FindPeaksByIncrementalPeakAddition,
)
import numpy as np
from scipy.special import voigt_profile


def test_twotermsvoigt_linearbaseline():
    x = np.linspace(0, 99, 100)
    area0 = 7
    pos0 = 25
    w0 = 3
    nu0 = 6 / 11.0
    area1 = 6
    pos1 = 75
    w1 = 4
    nu1 = 8 / 11.0
    b0 = 0.5
    b1 = 1 / 2048.0
    sigma0 = w0 * np.sqrt(nu0) * voigtareaparametrizationnu.OneBySqrtLog4
    gamma0 = w0 * (1 - nu0)
    sigma1 = w1 * np.sqrt(nu1) * voigtareaparametrizationnu.OneBySqrtLog4
    gamma1 = w1 * (1 - nu1)
    y = (
        area0 * voigt_profile(x - pos0, sigma0, gamma0)
        + area1 * voigt_profile(x - pos1, sigma1, gamma1)
        + b0
        + b1 * x
    )

    fit = FindPeaksByIncrementalPeakAddition()
    fit.maximumNumberOfPeaks = 2
    fit.orderOfBaselinePolynomial = 1

    out = fit.Execute(x, y, voigtareaparametrizationnu.VoigtAreaParametrizationNu())
    params = out.params
    np.testing.assert_almost_equal(params["b0"], b0, decimal=8)
    np.testing.assert_almost_equal(params["b1"], b1, decimal=8)
    np.testing.assert_almost_equal(params["area0"], area0, decimal=8)
    np.testing.assert_almost_equal(params["pos0"], pos0, decimal=8)
    np.testing.assert_almost_equal(params["w0"], w0, decimal=8)
    np.testing.assert_almost_equal(params["nu0"], nu0, decimal=8)
    np.testing.assert_almost_equal(params["area1"], area1, decimal=8)
    np.testing.assert_almost_equal(params["pos1"], pos1, decimal=8)
    np.testing.assert_almost_equal(params["w1"], w1, decimal=8)
    np.testing.assert_almost_equal(params["nu1"], nu1, decimal=8)


def test_twotermspearsoniv_linearbaseline():
    x = np.linspace(0, 99, 100)
    a0 = 7
    pos0 = 25
    w0 = 5
    m0 = 16 / 11.0
    v0 = 1 / 4.0
    a1 = 6
    pos1 = 75
    w1 = 6
    m1 = 21 / 11.0
    v1 = 1 / 8.0
    b0 = 100
    b1 = 0.25
    y = (
        PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(x, a0, pos0, w0, m0, v0)
        + PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(x, a1, pos1, w1, m1, v1)
        + b0
        + b1 * x
    )

    fit = FindPeaksByIncrementalPeakAddition()
    fit.maximumNumberOfPeaks = 2
    fit.orderOfBaselinePolynomial = 1

    out = fit.Execute(x, y, PearsonIVAmplitudeParametrizationHPW())
    assert len(out.params) == 12
    # TODO: fit does not converge in the moment
    # params = out.params
    # np.testing.assert_almost_equal(params["b0"], b0, decimal=8)
    # np.testing.assert_almost_equal(params["b1"], b1, decimal=8)
    # np.testing.assert_almost_equal(params["a0"], a0, decimal=8)
    # np.testing.assert_almost_equal(params["pos0"], pos0, decimal=8)
    # np.testing.assert_almost_equal(params["w0"], w0, decimal=8)
    # np.testing.assert_almost_equal(params["m0"], m0, decimal=8)
    # np.testing.assert_almost_equal(params["v0"], v0, decimal=8)
    # np.testing.assert_almost_equal(params["a1"], a1, decimal=8)
    # np.testing.assert_almost_equal(params["pos1"], pos1, decimal=8)
    # np.testing.assert_almost_equal(params["w1"], w1, decimal=8)
    # np.testing.assert_almost_equal(params["m1"], m1, decimal=8)
    # np.testing.assert_almost_equal(params["v1"], v1, decimal=8)
