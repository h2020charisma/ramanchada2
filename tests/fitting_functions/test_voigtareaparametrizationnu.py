from ramanchada2.fitting_functions import voigtareaparametrizationnu
import lmfit
import numpy as np
from scipy.special import voigt_profile


def test_empty_func():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(0, -1)
    p = lmfit.Parameters()
    x = np.array([3])
    y = ff.func(p, x)
    assert y[0] == 0, 'empty function must return 0'


def test_constantbaseline():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(0, 0)
    p = lmfit.Parameters()
    p.add('b0', 66)
    x = np.array([3])
    y = ff.func(p, x)
    assert y[0] == 66, 'const background must return parameter b0'


def test_linearbaseline():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(0, 1)
    p = lmfit.Parameters()
    p.add('b0', 66)
    p.add('b1', 7)
    x = np.array([3, 5])
    y = ff.func(p, x)
    assert y[0] == 66 + 7 * 3, 'linear background must return parameter b0 + b1 * x'
    assert y[1] == 66 + 7 * 5, 'linear background must return parameter b0 + b1 * x'


def test_oneterm_plus_linearbaseline():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(1, 1)
    p = lmfit.Parameters()
    p.add('b0', 4)
    p.add('b1', -1)
    p.add('area0', 17)
    p.add('pos0', 1)
    p.add('w0', 3)
    p.add('nu0', 5.0 / 7.0)
    x = np.array([3])
    y = ff.func(p, x)
    assert y[0] == 1.72519653836887579809275 + 4 - 1 * 3, 'test with one term and linear background'


def test_derivatives_generalcase():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(1, 1)
    area = 17
    position = 7
    w = 3
    nu = 5 / 11.0
    b0 = 6
    b1 = -1
    expectedFunctionValue = 1.53969430505912493288384
    expectedDerivativeWrtArea = 0.0905702532387720548755199
    expectedDerivativeWrtPosition = 0.469766282659418172420053
    expectedDerivativeWrtW = -0.200053913246762862681244
    expectedDerivativeWrtNu = 0.677517478495227285639865
    p = lmfit.Parameters()
    p.add('area0', area)
    p.add('pos0', position)
    p.add('w0', w)
    p.add('nu0', nu)
    p.add('b0', b0)
    p.add('b1', b1)
    x = np.array([9, 9])
    y = ff.func(p, x)
    np.testing.assert_almost_equal(y[0], expectedFunctionValue + b0 + b1 * x[0], 12)
    np.testing.assert_almost_equal(y[1], expectedFunctionValue + b0 + b1 * x[1], 12)
    dy = ff.dfunc(p, x)
    np.testing.assert_almost_equal(dy[0, 0], expectedDerivativeWrtArea, decimal=12)
    np.testing.assert_almost_equal(dy[1, 0], expectedDerivativeWrtPosition, decimal=12)
    np.testing.assert_almost_equal(dy[2, 0], expectedDerivativeWrtW, decimal=12)
    np.testing.assert_almost_equal(dy[3, 0], expectedDerivativeWrtNu, decimal=12)
    np.testing.assert_almost_equal(dy[4, 0], 1, decimal=12)
    np.testing.assert_almost_equal(dy[5, 0], x[0], decimal=12)


def test_derivatives_lorentzlimit():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(1, -1)
    area = 17
    position = 7
    w = 3
    nu = 1 / 16383.0
    expectedFunctionValue = 1.248792269568032230171214
    expectedDerivativeWrtArea = 0.0734561275808747703548694
    expectedDerivativeWrtPosition = 0.384232051961498798779317
    expectedDerivativeWrtW = -0.160096688317291166158049
    expectedDerivativeWrtNu = 0.624202576968383211902258
    p = lmfit.Parameters()
    p.add('area0', area)
    p.add('pos0', position)
    p.add('w0', w)
    p.add('nu0', nu)
    x = np.array([9, 9])
    y = ff.func(p, x)
    np.testing.assert_almost_equal(y[0], expectedFunctionValue, 12)
    np.testing.assert_almost_equal(y[1], expectedFunctionValue, 12)
    dy = ff.dfunc(p, x)
    np.testing.assert_almost_equal(dy[0, 0], expectedDerivativeWrtArea, decimal=12)
    np.testing.assert_almost_equal(dy[1, 0], expectedDerivativeWrtPosition, decimal=12)
    np.testing.assert_almost_equal(dy[2, 0], expectedDerivativeWrtW, decimal=12)
    np.testing.assert_almost_equal(dy[3, 0], expectedDerivativeWrtNu, decimal=12)


def test_derivatives_gausslimit():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(1, -1)
    area = 17
    position = 7
    w = 3
    nu = 1
    expectedFunctionValue = 1.95602477676540327048287
    expectedDerivativeWrtArea = 0.1150602809862001923813453
    expectedDerivativeWrtPosition = 0.602583581831260309934069
    expectedDerivativeWrtW = -0.250285871034294216871578
    expectedDerivativeWrtNu = 0.865084621539303204435295
    p = lmfit.Parameters()
    p.add('area0', area)
    p.add('pos0', position)
    p.add('w0', w)
    p.add('nu0', nu)
    x = np.array([9, 9])
    y = ff.func(p, x)
    np.testing.assert_almost_equal(y[0], expectedFunctionValue, 12)
    np.testing.assert_almost_equal(y[1], expectedFunctionValue, 12)
    dy = ff.dfunc(p, x)
    np.testing.assert_almost_equal(dy[0, 0], expectedDerivativeWrtArea, decimal=12)
    np.testing.assert_almost_equal(dy[1, 0], expectedDerivativeWrtPosition, decimal=12)
    np.testing.assert_almost_equal(dy[2, 0], expectedDerivativeWrtW, decimal=12)
    np.testing.assert_almost_equal(dy[3, 0], expectedDerivativeWrtNu, decimal=12)


def test_fit_without_derivatives_oneterm():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(1, -1)
    x = np.linspace(0, 99, 100)
    w = 5
    nu = 6 / 11.0
    sigma = w * np.sqrt(nu) * voigtareaparametrizationnu.OneBySqrtLog4
    gamma = w * (1 - nu)
    data = 7 * voigt_profile(x - 50, sigma, gamma)
    params = lmfit.Parameters()
    params.add('area0', 10)
    params.add('pos0', 49)
    params.add('w0', 6)
    params.add('nu0', 0.5)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={'data': data})
    out1 = min1.leastsq()
    np.testing.assert_almost_equal(out1.params['area0'], 7, 12)
    np.testing.assert_almost_equal(out1.params['pos0'], 50, 12)
    np.testing.assert_almost_equal(out1.params['w0'], 5, 12)
    np.testing.assert_almost_equal(out1.params['nu0'], 6 / 11.0, 12)


def test_fit_with_derivatives_oneterm():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(1, -1)
    x = np.linspace(0, 99, 100)
    area = 7
    pos = 50
    w = 5
    nu = 6 / 11.0
    sigma = w * np.sqrt(nu) * voigtareaparametrizationnu.OneBySqrtLog4
    gamma = w * (1 - nu)
    data = area * voigt_profile(x - pos, sigma, gamma)
    params = lmfit.Parameters()

    # important: the parameters must be added in exactly the order
    # as you can see here (area0, pos0, w0, nu0, ... areaN, posN, wN, nuN, b0, b1, ... bM)
    params.add('area0', 10)
    params.add('pos0', 49)
    params.add('w0', 6)
    params.add('nu0', 0.5)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={'data': data})
    out1 = min1.leastsq(Dfun=ff.dfunc, col_deriv=1)
    np.testing.assert_almost_equal(out1.params['area0'], area, 12)
    np.testing.assert_almost_equal(out1.params['pos0'], pos, 12)
    np.testing.assert_almost_equal(out1.params['w0'], w, 12)
    np.testing.assert_almost_equal(out1.params['nu0'], nu, 12)


def test_fit_with_derivatives_twoterms_linearbaseline():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(2, 1)
    x = np.linspace(0, 99, 100)
    area0 = 7
    pos0 = 25
    w0 = 3
    nu0 = 6 / 11.0
    area1 = 6
    pos1 = 75
    w1 = 4
    nu1 = 8 / 11.0
    b0 = 100
    b1 = 0.25
    sigma0 = w0 * np.sqrt(nu0) * voigtareaparametrizationnu.OneBySqrtLog4
    gamma0 = w0 * (1 - nu0)
    sigma1 = w1 * np.sqrt(nu1) * voigtareaparametrizationnu.OneBySqrtLog4
    gamma1 = w1 * (1 - nu1)
    data = area0 * voigt_profile(x - pos0, sigma0, gamma0) + area1 * voigt_profile(x - pos1, sigma1, gamma1) + b0 + b1 * x

    # important: the parameters must be added in exactly the order
    # as you can see here (area0, pos0, w0, nu0, ... areaN, posN, wN, nuN, b0, b1, ... bM)
    params = lmfit.Parameters()
    params.add('area0', 10)
    params.add('pos0', 24)
    params.add('w0', 3.5)
    params.add('nu0', 0.5)
    params.add('area1', 10)
    params.add('pos1', 74)
    params.add('w1', 4.5)
    params.add('nu1', 0.5)
    params.add('b0', 99)
    params.add('b1', 0.24)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={'data': data})
    out1 = min1.leastsq(Dfun=ff.dfunc, col_deriv=1)

    np.testing.assert_almost_equal(out1.params['area0'], area0, 12)
    np.testing.assert_almost_equal(out1.params['pos0'], pos0, 12)
    np.testing.assert_almost_equal(out1.params['w0'], w0, 12)
    np.testing.assert_almost_equal(out1.params['nu0'], nu0, 12)
    np.testing.assert_almost_equal(out1.params['area1'], area1, 12)
    np.testing.assert_almost_equal(out1.params['pos1'], pos1, 12)
    np.testing.assert_almost_equal(out1.params['w1'], w1, 12)
    np.testing.assert_almost_equal(out1.params['nu1'], nu1, 12)
    np.testing.assert_almost_equal(out1.params['b0'], b0, 12)
    np.testing.assert_almost_equal(out1.params['b1'], b1, 12)
