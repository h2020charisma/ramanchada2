from ramanchada2.fitting_functions.pearsonivamplitudeparametrizationhpw import (
    PearsonIVAmplitudeParametrizationHPW,
    )
from ramanchada2.fitting_functions.models import PearsonIVParametrizationHPWModel
import lmfit
import numpy as np


def almostequal(desired, actual, abstol, reltol):
    delta = np.abs(abstol) + np.abs(reltol * desired)
    return np.abs(desired - actual) <= delta


def test_FunctionValuesBasics():
    # amp, pos, w, m, v
    testData = [
        (3, 7, 5, 11, 13),
        (3, 7, 1 / 5.0, 1 / 1000.0, 1000),
        (3, 7, 1 / 5.0, 1 / 1000.0, -1000),
        (3, 7, 1 / 5.0, 1 / 1000.0, 0),
        (3, 7, 1 / 5.0, 1000.0, 1000),
        (3, 7, 1 / 5.0, 1000.0, -1000),
        (3, 7, 1 / 5.0, 1000.0, 0),
    ]

    func = PearsonIVAmplitudeParametrizationHPW(1, -1)
    for (amp, pos, w, m, v) in testData:
        p = lmfit.Parameters()
        p.add("a0", amp)
        p.add("pos0", pos)
        p.add("w0", w)
        p.add("m0", m)
        p.add("v0", v)

        # Y at x=pos should be equal to amp
        X = np.array([pos])
        y = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(X[0], amp, pos, w, m, v)
        assert almostequal(amp, y, 1e-10, 1e-10)
        Y = func.func(p, X)
        assert almostequal(y, Y[0], 1e-15, 1e-15)

        # Y at x= pos - w /1000 should be less than amp
        X = np.array([pos - w / 1000.0])
        y = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(X[0], amp, pos, w, m, v)
        assert amp > y
        Y = func.func(p, X)
        assert almostequal(y, Y[0], 1e-15, 1e-15)

        # Y at x= pos + w /1000 should be less than amp
        X = np.array([pos + w / 1000.0])
        y = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(X[0], amp, pos, w, m, v)
        assert amp > y
        Y = func.func(p, X)
        assert almostequal(y, Y[0], 1e-15, 1e-15)

        # Y at x= pos -3 w should be less than amp/2
        X = np.array([pos - 3 * w])
        y = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(X[0], amp, pos, w, m, v)
        assert amp / 2 > y
        Y = func.func(p, X)
        assert almostequal(y, Y[0], 1e-15, 1e-15)

        # Y at x= pos + 3 w should be less than amp/2
        X = np.array([pos + 3 * w])
        y = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(X[0], amp, pos, w, m, v)
        assert amp / 2 > y
        Y = func.func(p, X)
        assert almostequal(y, Y[0], 1e-15, 1e-15)


def test_FunctionValuesInDependenceOnX():
    testData = [
        (-1e9, 3, 7, 5, 11, 13, 2.7867315365921926531e-160),
        (-19, 3, 7, 5, 11, 13, 0.0071554075200805234436),
        (-1, 3, 7, 5, 11, 13, 0.94609302487197904495),
        (0, 3, 7, 5, 11, 13, 1.1845028216303614895),
        (6, 3, 7, 5, 11, 13, 2.9207184451185306856),
        (7, 3, 7, 5, 11, 13, 3.0000000000000000000),
        (8, 3, 7, 5, 11, 13, 2.9093699544835131623),
        (23, 3, 7, 5, 11, 13, 5.7924047779440271306e-25),
        (28, 3, 7, 5, 11, 13, 7.9007462868389145363e-260),
    ]

    func = PearsonIVAmplitudeParametrizationHPW(1, -1)
    for (x, amp, pos, w, m, v, yexpected) in testData:
        p = lmfit.Parameters()
        p.add("a0", amp)
        p.add("pos0", pos)
        p.add("w0", w)
        p.add("m0", m)
        p.add("v0", v)

        X = np.array([x])
        y = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(X[0], amp, pos, w, m, v)
        if yexpected < 1e-100:
            assert almostequal(yexpected, y, 0, 1e-7)
        else:
            assert almostequal(yexpected, y, 0, 1e-13)

        Y = func.func(p, X)
        assert almostequal(y, Y[0], 1e-15, 1e-15)


def test_DerivativesWrtParameters():
    ff = PearsonIVAmplitudeParametrizationHPW(1, -1)

    # General case
    amp = 17
    pos = 7
    w = 3
    m = 5
    v = 7

    expectedFunctionValue = 10.456864176583425249219583024250
    expectedDerivativeWrtAmplitude = 0.61510965744608383818938723672062
    expectedDerivativeWrtPosition = 6.2023944308469781499266683603126
    expectedDerivativeWrtW = 4.1349296205646520999511122402084
    expectedDerivativeWrtM = 0.31213582841409496793210643128803
    expectedDerivativeWrtV = -0.0080620681798224796387712054749779

    p = lmfit.Parameters()
    p.add("a0", amp)
    p.add("pos0", pos)
    p.add("w0", w)
    p.add("m0", m)
    p.add("v0", v)
    X = np.array([9])
    FV = ff.func(p, X)
    DY = ff.dfunc(p, X)

    assert almostequal(expectedFunctionValue, FV[0], 0, 1e-12)
    assert almostequal(expectedDerivativeWrtAmplitude, DY[0, 0], 0, 1e-12)
    assert almostequal(expectedDerivativeWrtPosition, DY[1, 0], 0, 1e-12)
    assert almostequal(expectedDerivativeWrtW, DY[2, 0], 0, 1e-12)
    assert almostequal(expectedDerivativeWrtM, DY[3, 0], 0, 1e-12)
    assert almostequal(expectedDerivativeWrtV, DY[4, 0], 0, 1e-11)


def test_AreaFwhm():
    testData = [
        (1, 7, 5, 11, 13),
        (3, 7, 5, 1000, -1000),
        (3, 7, 5, 1000, 0),
        (3, 7, 5, 1000, 1000),
        (3, 7, 5, 513 / 1024.0, -1000),
        (3, 7, 5, 513 / 1024.0, 0),
        (3, 7, 5, 513 / 1024.0, 1000),
    ]

    expectedAreas = [
        11.017024491957193080,
        31.945780266160134464,
        31.940456418616702875,
        31.945780266160134464,
        12084.703859367018981,
        8896.1196272036681069,
        12084.703859367018981,
    ]

    expectedFwhm = [
        10.069306246733044125,
        10.000770127169960848,
        10.000000000000000000,
        10.000770127169960848,
        11.364892411874591570,
        10.000000000000000000,
        11.364892411874591570,
    ]

    func = PearsonIVAmplitudeParametrizationHPW(1, -1)

    for i in range(len(testData)):
        (amp, pos, w, m, v) = testData[i]
        p = lmfit.Parameters()
        p.add("a0", amp)
        p.add("pos0", pos)
        p.add("w0", w)
        p.add("m0", m)
        p.add("v0", v)
        (
            position,
            posErr,
            area,
            areaErr,
            height,
            heightErr,
            fwhm,
            fwhmErr,
        ) = func.GetPositionAreaHeightFWHMFromSinglePeakParameters(p, 0)

        assert almostequal(pos, position, 0, 1e-14)
        assert almostequal(amp, height, 0, 1e-14)
        assert almostequal(expectedAreas[i], area, 0, 1e-8)
        assert almostequal(expectedFwhm[i], fwhm, 0, 1e-8)

    # with the first test set, we test additionally
    (amp, pos, w, m, v) = testData[0]
    p = lmfit.Parameters()
    p.add("a0", amp)
    p.add("pos0", pos)
    p.add("w0", w)
    p.add("m0", m)
    p.add("v0", v)
    (
        position,
        posErr,
        area,
        areaErr,
        height,
        heightErr,
        fwhm,
        fwhmErr,
    ) = func.GetPositionAreaHeightFWHMFromSinglePeakParameters(p, 0)

    # (i) when we vary the amplitude, area and height should change accordingly, pos and fwhm stay constant
    for newAmp in [-11, 0, 2, 17, 33, 1023, -11]:
        p = lmfit.Parameters()
        p.add("a0", newAmp)
        p.add("pos0", pos)
        p.add("w0", w)
        p.add("m0", m)
        p.add("v0", v)
        (
            newposition,
            _,
            newarea,
            _,
            newheight,
            _,
            newfwhm,
            _,
        ) = func.GetPositionAreaHeightFWHMFromSinglePeakParameters(p, 0)
        assert almostequal(position, newposition, 0, 1e-14)
        assert almostequal(height * newAmp, newheight, 0, 1e-14)
        assert almostequal(area * newAmp, newarea, 0, 1e-14)
        assert almostequal(fwhm, newfwhm, 0, 1e-14)

    # (ii) when we vary the position, position should change accordingly, everything else is the same
    for newPos in [-11, 0, 2, 17, 33, 1023, -11]:
        p = lmfit.Parameters()
        p.add("a0", amp)
        p.add("pos0", newPos)
        p.add("w0", w)
        p.add("m0", m)
        p.add("v0", v)
        (
            newposition,
            _,
            newarea,
            _,
            newheight,
            _,
            newfwhm,
            _,
        ) = func.GetPositionAreaHeightFWHMFromSinglePeakParameters(p, 0)
        assert almostequal(newPos, newposition, 0, 1e-14)
        assert almostequal(height, newheight, 0, 1e-14)
        assert almostequal(area, newarea, 0, 1e-14)
        assert almostequal(fwhm, newfwhm, 0, 1e-14)

    # (iii) when we vary the width, area and fwhm should change accordingly, everything else is the same
    for newW in [1 / 1000.0, 2, 17, 33, 1023]:
        p = lmfit.Parameters()
        p.add("a0", amp)
        p.add("pos0", pos)
        p.add("w0", newW)
        p.add("m0", m)
        p.add("v0", v)
        (
            newposition,
            _,
            newarea,
            _,
            newheight,
            _,
            newfwhm,
            _,
        ) = func.GetPositionAreaHeightFWHMFromSinglePeakParameters(p, 0)
        assert almostequal(position, newposition, 0, 1e-14)
        assert almostequal(height, newheight, 0, 1e-14)
        assert almostequal(area * (newW / w), newarea, 0, 1e-14)
        assert almostequal(fwhm * (newW / w), newfwhm, 0, 1e-14)


def test_FwhmApproximation():
    claimedMaxApproximationError = 0.19  # 19% error
    w = 7
    for idx_v in range(-1, 16):
        v = 0 if idx_v < 0 else np.power(10, (idx_v - 7.5) / 2.5)
        for idx_m in range(0, 16):
            m = np.power(10, (idx_m - 7.5) / 2.5)
            fwhmP = PearsonIVAmplitudeParametrizationHPW.GetFWHM(w, m, v)
            fwhmN = PearsonIVAmplitudeParametrizationHPW.GetFWHM(w, m, -v)

            fwhmApproxP = PearsonIVAmplitudeParametrizationHPW.GetFWHMApproximation(
                w, m, v
            )
            fwhmApproxN = PearsonIVAmplitudeParametrizationHPW.GetFWHMApproximation(
                w, m, -v
            )

            # FWHM should be independent of whether v is positive or negative
            assert almostequal(fwhmP, fwhmN, 1e-8, 1e-8)

            assert almostequal(fwhmP, fwhmApproxP, 0, claimedMaxApproximationError)
            assert almostequal(fwhmN, fwhmApproxN, 0, claimedMaxApproximationError)


def test_DerivativesOfAreaFwhm():
    # see PearsonIV (amplitudeModified).nb
    amp = 3
    pos = 7
    w = 5
    m = 11
    v = 13

    ymaxDerivs = [1, 0, 0, 0, 0]
    xmaxDerivs = [0, 1, 0, 0, 0]
    areaDerivs = [
        11.0170244919571930798,
        0,
        6.6102146951743158479,
        -0.106359067713399883296,
        0.00045052750734765666701,
    ]
    fwhmDerivs = [
        0,
        0,
        2.0138612493466088250930420306,
        -0.0062732720000624223737773164,
        0.0000628521667333096080982645,
    ]

    p = lmfit.Parameters()
    p.add("a0", amp)
    p.add("pos0", pos)
    p.add("w0", w)
    p.add("m0", m)
    p.add("v0", v)

    func = PearsonIVAmplitudeParametrizationHPW(1, -1)

    for i in range(0, 5):
        cov = np.zeros((5, 5))
        cov[i, i] = 1
        result = func.GetPositionAreaHeightFWHMFromSinglePeakParameters(p, 0, cov)
        assert almostequal(np.abs(ymaxDerivs[i]), result.HeightStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(xmaxDerivs[i]), result.PositionStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(areaDerivs[i]), result.AreaStdDev, 1e-13, 1e-4)
        assert almostequal(np.abs(fwhmDerivs[i]), result.FWHMStdDev, 1e-13, 1e-4)


def test_fit_without_derivatives_oneterm():
    ff = PearsonIVAmplitudeParametrizationHPW(1, -1)
    x = np.linspace(0, 99, 100)
    a = 7
    pos = 50
    w = 5
    m = 16 / 11.0
    v = 1 / 4.0
    data = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(x, a, pos, w, m, v)
    params = lmfit.Parameters()
    params.add("a0", 10)
    params.add("pos0", 49)
    params.add("w0", 6)
    params.add("m0", 1)
    params.add("v0", 0)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={"data": data})
    out1 = min1.leastsq()
    np.testing.assert_almost_equal(out1.params["a0"], a, 12)
    np.testing.assert_almost_equal(out1.params["pos0"], pos, 12)
    np.testing.assert_almost_equal(out1.params["w0"], w, 12)
    np.testing.assert_almost_equal(out1.params["m0"], m, 12)
    np.testing.assert_almost_equal(out1.params["v0"], v, 12)


def test_fit_with_derivatives_oneterm():
    ff = PearsonIVAmplitudeParametrizationHPW(1, -1)
    x = np.linspace(0, 99, 100)
    a = 7
    pos = 50
    w = 5
    m = 16 / 11.0
    v = 1 / 4.0
    data = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(x, a, pos, w, m, v)
    params = lmfit.Parameters()
    # important: the parameters must be added in exactly the order
    # as you can see here (area0, pos0, w0, nu0, ... areaN, posN, wN, nuN, b0, b1, ... bM)
    params.add("a0", 10)
    params.add("pos0", 49)
    params.add("w0", 6)
    params.add("m0", 1)
    params.add("v0", 0)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={"data": data})
    out1 = min1.leastsq(Dfun=ff.dfunc, col_deriv=1)
    np.testing.assert_almost_equal(out1.params["a0"], a, 12)
    np.testing.assert_almost_equal(out1.params["pos0"], pos, 12)
    np.testing.assert_almost_equal(out1.params["w0"], w, 12)
    np.testing.assert_almost_equal(out1.params["m0"], m, 12)
    np.testing.assert_almost_equal(out1.params["v0"], v, 12)


def test_fit_with_derivatives_twoterms_linearbaseline():
    ff = PearsonIVAmplitudeParametrizationHPW(2, 1)
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
    data = (
        PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(x, a0, pos0, w0, m0, v0)
        + PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(x, a1, pos1, w1, m1, v1)
        + b0
        + b1 * x
    )

    # important: the parameters must be added in exactly the order
    # as you can see here (b0, b1, ... bM, area0, pos0, w0, nu0, ... areaN, posN, wN, nuN, )
    params = lmfit.Parameters()
    params.add("b0", 99)
    params.add("b1", 0.24)
    params.add("a0", 10)
    params.add("pos0", 24)
    params.add("w0", 6)
    params.add("m0", 1)
    params.add("v0", 0)
    params.add("a1", 10)
    params.add("pos1", 74)
    params.add("w1", 7)
    params.add("m1", 1)
    params.add("v1", 0)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={"data": data})
    out1 = min1.leastsq(Dfun=ff.dfunc, col_deriv=1)
    cv = out1.covar
    print(np.shape(cv))

    np.testing.assert_almost_equal(out1.params["b0"], b0, 12)
    np.testing.assert_almost_equal(out1.params["b1"], b1, 12)
    np.testing.assert_almost_equal(out1.params["a0"], a0, 12)
    np.testing.assert_almost_equal(out1.params["pos0"], pos0, 12)
    np.testing.assert_almost_equal(out1.params["w0"], w0, 12)
    np.testing.assert_almost_equal(out1.params["m0"], m0, 12)
    np.testing.assert_almost_equal(out1.params["v0"], v0, 12)
    np.testing.assert_almost_equal(out1.params["a1"], a1, 12)
    np.testing.assert_almost_equal(out1.params["pos1"], pos1, 12)
    np.testing.assert_almost_equal(out1.params["w1"], w1, 12)
    np.testing.assert_almost_equal(out1.params["m1"], m1, 12)
    np.testing.assert_almost_equal(out1.params["v1"], v1, 12)


def test_fit_with_model_oneterm():
    x = np.linspace(0, 99, 100)
    a = 7
    pos = 50
    w = 5
    m = 16 / 11.0
    v = 1 / 4.0
    data = PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm(x, a, pos, w, m, v)
    model = PearsonIVParametrizationHPWModel(x=x)
    params = model.guess(data, x)
    result = model.fit(data, x=x, params=params)
    np.testing.assert_almost_equal(result.params["height"], a, 7)
    np.testing.assert_almost_equal(result.params["center"], pos, 7)
    np.testing.assert_almost_equal(result.params["sigma"], w, 7)
    np.testing.assert_almost_equal(result.params["expon"], m, 7)
    np.testing.assert_almost_equal(result.params["skew"], v, 7)
