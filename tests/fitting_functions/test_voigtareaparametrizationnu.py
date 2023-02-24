from ramanchada2.fitting_functions import voigtareaparametrizationnu
from ramanchada2.fitting_functions.voigtareaparametrizationnu import (
    VoigtAreaParametrizationNu,
)
import lmfit
import numpy as np
from scipy.special import voigt_profile

# The voigt HWHM with sigma=1, and gamma = (1+d)/(1-d), d = [-1, 1/64, 1)
voigt_hwhm_sigmaone_dvar = [
    1.177410022515474691,
    1.1816092691813030468,
    1.1858873739520897034,
    1.1902465142050570863,
    1.194688946189857397,
    1.1992170085518593889,
    1.2038331260421231307,
    1.2085398134254873464,
    1.2133396795989858683,
    1.2182354319336629671,
    1.2232298808537770661,
    1.2283259446683743323,
    1.2335266546712841194,
    1.2388351605267440826,
    1.2442547359591114981,
    1.2497887847664671425,
    1.2554408471793780475,
    1.2612146065876654677,
    1.2671138966597353836,
    1.2731427088808827742,
    1.2793052005389909397,
    1.2856057031882278137,
    1.2920487316237085011,
    1.2986389934026648102,
    1.3053813989504578017,
    1.3122810722928098214,
    1.3193433624589418804,
    1.3265738556039068772,
    1.3339783879023381003,
    1.3415630592701179674,
    1.3493342479751497996,
    1.3572986262035252986,
    1.3654631766529664197,
    1.3738352102315315844,
    1.3824223849462672922,
    1.3912327260738180846,
    1.4002746477130484284,
    1.4095569758285553194,
    1.4190889729036451368,
    1.4288803643320075071,
    1.4389413666890491483,
    1.4492827180367713744,
    1.4599157104303203962,
    1.4708522248100608452,
    1.4821047684803902488,
    1.4936865153957175192,
    1.5056113494952888279,
    1.5178939113521050052,
    1.5305496484273140468,
    1.5435948692504962536,
    1.5570468018785470874,
    1.5709236570218130826,
    1.5852446962662159034,
    1.6000303058648426117,
    1.6153020766224977944,
    1.6310828904527072039,
    1.6473970142494402595,
    1.6642702017863107582,
    1.681729804435296653,
    1.69980489158633592,
    1.7185263817499517539,
    1.7379271854390185723,
    1.7580423610548628274,
    1.7789092851493945634,
    1.8005678386015786904,
    1.823060610436444369,
    1.8464331212317292609,
    1.8707340683055766275,
    1.8960155951636819839,
    1.9223335880121482867,
    1.9497480025204815182,
    1.9783232244565282241,
    2.008128468322387214,
    2.0392382187102560495,
    2.0717327197852989941,
    2.1056985191077205408,
    2.141229072951108682,
    2.1784254213865807416,
    2.2173969427163275031,
    2.2582621983975355419,
    2.3011498814497513326,
    2.3461998835490209547,
    2.393564498659336335,
    2.4434097842340786489,
    2.4959171048598266576,
    2.551284887865954588,
    2.6097306260799226329,
    2.6714931698159537872,
    2.7368353586570783274,
    2.8060470540283742602,
    2.8794486464792013645,
    2.9573951276637198642,
    3.0402808371050941667,
    3.1285450190960804891,
    3.2226783570437941673,
    3.3232306932306941103,
    3.4308201940528437269,
    3.5461442879894212769,
    3.6699927908828172788,
    3.8032637475035756812,
    3.9469826695032671633,
    4.1023260513255730736,
    4.2706503168149816862,
    4.4535277180208407742,
    4.6527912147146390367,
    4.8705910685478533488,
    5.1094668797693460142,
    5.3724402144314312228,
    5.6631350290659021695,
    5.9859361342534139213,
    6.3462004902284115273,
    6.7505430891683662786,
    7.2072300562145225172,
    7.7267290051348977324,
    8.3224952766020927734,
    9.0121210735938730413,
    9.8190591847972300101,
    10.775286943841163681,
    11.925568590252918077,
    13.334559234025861601,
    15.0992368270188772,
    17.371990147302994501,
    20.406794809689181251,
    24.660800535533617726,
    31.04829937949178548,
    41.702630469368096869,
    63.023799037287215586,
    127.01180974246945707,
]

# The voigt HWHM with gamma=1, and sigma = (1+d)/(1-d), d = [-1, 1/64, 1)
voigt_hwhm_gammaone_dvar = [
    1,
    1.0000929900981847014,
    1.0003777624966224696,
    1.0008631312648343249,
    1.0015580444997350155,
    1.0024715664851064116,
    1.0036128594929105533,
    1.0049911655464542273,
    1.0066157884679251467,
    1.0084960765229643227,
    1.0106414059536371252,
    1.0130611656602803461,
    1.0157647432548858631,
    1.0187615126671334742,
    1.0220608234423622704,
    1.0256719918320660707,
    1.0296042937449317882,
    1.033866959602362403,
    1.0384691711282855227,
    1.0434200600992189404,
    1.0487287090862781795,
    1.0544041542342061278,
    1.0604553901408076633,
    1.0668913769200059716,
    1.0737210495495320854,
    1.0809533296167089258,
    1.0885971395802894494,
    1.0966614196612918117,
    1.1051551474609148057,
    1.1140873603798353005,
    1.1234671808824950853,
    1.133303844615175872,
    1.143606731350947909,
    1.1543853987011884804,
    1.1656496185052021456,
    1.1774094157888474959,
    1.189675110171558587,
    1.2024573595995344503,
    1.2157672062912183539,
    1.2296161247989505185,
    1.2440160721168537852,
    1.2589795397983230491,
    1.2745196080855436114,
    1.2906500020968946739,
    1.3073851501646711064,
    1.3247402444642595085,
    1.3427313041259691636,
    1.3613752410716541342,
    1.3806899288698507996,
    1.4006942749554334374,
    1.4214082966130304507,
    1.4428532011781249068,
    1.4650514709665480456,
    1.4880269535027891822,
    1.5118049576811641308,
    1.5364123565625216811,
    1.5618776975840789442,
    1.5882313210425649123,
    1.6155054878026846865,
    1.6437345172857499843,
    1.6729549369091311623,
    1.7032056442782115564,
    1.7345280835813214269,
    1.7669664378076306961,
    1.8005678386015786904,
    1.8353825957890578829,
    1.8714644488648539776,
    1.9088708430231843335,
    1.9476632326499453211,
    1.9879074155840199742,
    2.0296739019046683743,
    2.0730383215232993655,
    2.1180818754635660479,
    2.1648918364190477434,
    2.2135621050012006813,
    2.2641938290540225637,
    2.3168960945429309357,
    2.3717866978564628894,
    2.4289930109305334564,
    2.4886529524650857967,
    2.5509160807121900781,
    2.6159448259472447961,
    2.6839158838829061716,
    2.7550217950632123133,
    2.8294727398261995659,
    2.9074985839268644614,
    2.989351216595417954,
    3.0753072309560758432,
    3.1656710067159081262,
    3.2607782673217607212,
    3.3610001989823174293,
    3.4667482378486090287,
    3.5784796552666793171,
    3.6967041007104309105,
    3.8219912995573272195,
    3.9549801506665303188,
    4.0963895299588992592,
    4.247031185217482386,
    4.4078252100521560122,
    4.579818719577299268,
    4.7642085282226360724,
    4.9623688672590590593,
    5.1758854988773873769,
    5.406598017846376464,
    5.6566527287853171409,
    5.9285693177078176118,
    6.2253257069142318691,
    6.5504671543400179081,
    6.9082480829105510743,
    7.3038186983166432837,
    7.7434738129206051218,
    8.2349894900724039365,
    8.7880859302556463323,
    9.4150755119073858072,
    10.131788564238479342,
    10.958926420044274577,
    11.924090995155746488,
    13.064921411472708807,
    14.43411259407456938,
    16.107779600011765899,
    18.200095193984788025,
    20.89047391778342413,
    24.477940229523170324,
    29.500738410375740968,
    37.035357331885579308,
    49.593604758544045262,
    74.710904558981651312,
    150.06437718602548694,
]


def almostequal(desired, actual, abstol, reltol):
    delta = np.abs(abstol) + np.abs(reltol * desired)
    return np.abs(desired - actual) <= delta


def test_empty_func():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(0, -1)
    p = lmfit.Parameters()
    x = np.array([3])
    y = ff.func(p, x)
    assert y[0] == 0, "empty function must return 0"


def test_constantbaseline():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(0, 0)
    p = lmfit.Parameters()
    p.add("b0", 66)
    x = np.array([3])
    y = ff.func(p, x)
    assert y[0] == 66, "const background must return parameter b0"


def test_linearbaseline():
    ff = voigtareaparametrizationnu.VoigtAreaParametrizationNu(0, 1)
    p = lmfit.Parameters()
    p.add("b0", 66)
    p.add("b1", 7)
    x = np.array([3, 5])
    y = ff.func(p, x)
    assert y[0] == 66 + 7 * 3, "linear background must return parameter b0 + b1 * x"
    assert y[1] == 66 + 7 * 5, "linear background must return parameter b0 + b1 * x"


def test_oneterm_plus_linearbaseline():
    ff = VoigtAreaParametrizationNu(1, 1)
    p = lmfit.Parameters()
    p.add("b0", 4)
    p.add("b1", -1)
    p.add("area0", 17)
    p.add("pos0", 1)
    p.add("w0", 3)
    p.add("nu0", 5.0 / 7.0)
    x = np.array([3])
    y = ff.func(p, x)
    assert (
        y[0] == 1.72519653836887579809275 + 4 - 1 * 3
    ), "test with one term and linear background"


def test_derivatives_generalcase():
    ff = VoigtAreaParametrizationNu(1, 1)
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
    p.add("area0", area)
    p.add("pos0", position)
    p.add("w0", w)
    p.add("nu0", nu)
    p.add("b0", b0)
    p.add("b1", b1)
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
    ff = VoigtAreaParametrizationNu(1, -1)
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
    p.add("area0", area)
    p.add("pos0", position)
    p.add("w0", w)
    p.add("nu0", nu)
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
    ff = VoigtAreaParametrizationNu(1, -1)
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
    p.add("area0", area)
    p.add("pos0", position)
    p.add("w0", w)
    p.add("nu0", nu)
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
    ff = VoigtAreaParametrizationNu(1, -1)
    x = np.linspace(0, 99, 100)
    w = 5
    nu = 6 / 11.0
    sigma = w * np.sqrt(nu) * voigtareaparametrizationnu.OneBySqrtLog4
    gamma = w * (1 - nu)
    data = 7 * voigt_profile(x - 50, sigma, gamma)
    params = lmfit.Parameters()
    params.add("area0", 10)
    params.add("pos0", 49)
    params.add("w0", 6)
    params.add("nu0", 0.5)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={"data": data})
    out1 = min1.leastsq()
    np.testing.assert_almost_equal(out1.params["area0"], 7, 12)
    np.testing.assert_almost_equal(out1.params["pos0"], 50, 12)
    np.testing.assert_almost_equal(out1.params["w0"], 5, 12)
    np.testing.assert_almost_equal(out1.params["nu0"], 6 / 11.0, 12)


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
    params.add("area0", 10)
    params.add("pos0", 49)
    params.add("w0", 6)
    params.add("nu0", 0.5)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={"data": data})
    out1 = min1.leastsq(Dfun=ff.dfunc, col_deriv=1)
    np.testing.assert_almost_equal(out1.params["area0"], area, 12)
    np.testing.assert_almost_equal(out1.params["pos0"], pos, 12)
    np.testing.assert_almost_equal(out1.params["w0"], w, 12)
    np.testing.assert_almost_equal(out1.params["nu0"], nu, 12)


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
    data = (
        area0 * voigt_profile(x - pos0, sigma0, gamma0)
        + area1 * voigt_profile(x - pos1, sigma1, gamma1)
        + b0
        + b1 * x
    )

    # important: the parameters must be added in exactly the order
    # as you can see here (area0, pos0, w0, nu0, ... areaN, posN, wN, nuN, b0, b1, ... bM)
    params = lmfit.Parameters()
    params.add("area0", 10)
    params.add("pos0", 24)
    params.add("w0", 3.5)
    params.add("nu0", 0.5)
    params.add("area1", 10)
    params.add("pos1", 74)
    params.add("w1", 4.5)
    params.add("nu1", 0.5)
    params.add("b0", 99)
    params.add("b1", 0.24)

    min1 = lmfit.Minimizer(ff.func, params, fcn_args=(x,), fcn_kws={"data": data})
    out1 = min1.leastsq(Dfun=ff.dfunc, col_deriv=1)
    cv = out1.covar
    print(np.shape(cv))

    np.testing.assert_almost_equal(out1.params["area0"], area0, 12)
    np.testing.assert_almost_equal(out1.params["pos0"], pos0, 12)
    np.testing.assert_almost_equal(out1.params["w0"], w0, 12)
    np.testing.assert_almost_equal(out1.params["nu0"], nu0, 12)
    np.testing.assert_almost_equal(out1.params["area1"], area1, 12)
    np.testing.assert_almost_equal(out1.params["pos1"], pos1, 12)
    np.testing.assert_almost_equal(out1.params["w1"], w1, 12)
    np.testing.assert_almost_equal(out1.params["nu1"], nu1, 12)
    np.testing.assert_almost_equal(out1.params["b0"], b0, 12)
    np.testing.assert_almost_equal(out1.params["b1"], b1, 12)


def test_voigt_hwhm_exact_vargamma():
    sigma = 1
    for i in range(128):
        d = -1 + i / 64.0
        gamma = (1 + d) / (1 - d)
        hwhm = VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGamma(
            sigma, gamma
        )
        np.testing.assert_almost_equal(voigt_hwhm_sigmaone_dvar[i], hwhm, 0, 14)
        i += 1


def test_voigt_hwhm_Exact_varsigma():
    gamma = 1
    for i in range(128):
        d = -1 + i / 64.0
        sigma = (1 + d) / (1 - d)
        hwhm = VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGamma(
            sigma, gamma
        )
        np.testing.assert_almost_equal(voigt_hwhm_gammaone_dvar[i], hwhm, 0, 1e-14)
        i += 1


def test_voigt_hwhm_approx_vargamma():
    sigma = 1
    for i in range(128):
        d = -1 + i / 64.0
        gamma = (1 + d) / (1 - d)
        hwhm = VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGammaApproximation(
            sigma, gamma
        )
        assert (
            np.abs(1 - voigt_hwhm_sigmaone_dvar[i] / hwhm)
            < voigtareaparametrizationnu.VoigtHalfWidthHalfMaximumApproximationMaximalRelativeError
        )
        i += 1


def test_voigt_hwhm_approx_varsigma():
    gamma = 1
    for i in range(128):
        d = -1 + i / 64.0
        sigma = (1 + d) / (1 - d)
        hwhm = VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGammaApproximation(
            sigma, gamma
        )
        assert (
            np.abs(1 - hwhm / voigt_hwhm_gammaone_dvar[i])
            < voigtareaparametrizationnu.VoigtHalfWidthHalfMaximumApproximationMaximalRelativeError
        )
        i += 1


def test_derivatives_SecondaryParameters_ExactlyGaussianLimit():
    # see VoigtArea-Derivatives-ParametrizationSqrtNuLog4.nb
    area = 17
    pos = 7
    w = 3
    nu = 1

    # position, area, height and FWHM
    pahf = [pos, area, 2.66173895631567877902163, 6]
    # derivatives
    ymaxDerivs = [
        0.156572879783275222295390,
        0,
        -0.887246318771892926340544,
        1.16966732357221200231569,
    ]
    xmaxDerivs = [0, 1, 0, 0]
    areaDerivs = [1, 0, 0, 0]
    fwhmDerivs = [0, 0, 2, -0.207001611171248813604190]

    params = lmfit.Parameters()
    params.add("area0", area)
    params.add("pos0", pos)
    params.add("w0", w)
    params.add("nu0", nu)

    for i in range(4):
        cov = np.zeros((4, 4))
        cov[i, i] = 1
        result = VoigtAreaParametrizationNu.GetPositionAreaHeightFWHMFromSinglePeakParameters(
            params, 0, cov
        )
        assert almostequal(np.abs(areaDerivs[i]), result.AreaStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(xmaxDerivs[i]), result.PositionStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(ymaxDerivs[i]), result.HeightStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(fwhmDerivs[i]), result.FWHMStdDev, 1e-13, 1e-7)

        assert almostequal(pahf[0], result.Position, 1e-13, 1e-10)
        assert almostequal(pahf[1], result.Area, 1e-13, 1e-10)
        assert almostequal(pahf[2], result.Height, 1e-13, 1e-10)
        assert almostequal(pahf[3], result.FWHM, 1e-13, 1e-10)


def test_derivatives_SecondaryParameters_NearlyGaussianLimit():
    # see VoigtArea-Derivatives-ParametrizationSqrtNuLog4.nb
    area = 17
    pos = 7
    w = 3
    nu = 1 - 1 / 32767.0

    # position, area, height and FWHM
    pahf = [pos, area, 2.66170326013146224987024, 6.00000595967162241141443]
    # derivatives
    ymaxDerivs = [
        0.156570780007733073521779,
        0,
        -0.887234420043820749956747,
        1.16964641292668431368984,
    ]
    xmaxDerivs = [0, 1, 0, 0]
    areaDerivs = [1, 0, 0, 0]
    fwhmDerivs = [0, 0, 2.00000210576251930356528, -0.206995511602332887432860]

    params = lmfit.Parameters()
    params.add("area0", area)
    params.add("pos0", pos)
    params.add("w0", w)
    params.add("nu0", nu)

    for i in range(4):
        cov = np.zeros((4, 4))
        cov[i, i] = 1
        result = VoigtAreaParametrizationNu.GetPositionAreaHeightFWHMFromSinglePeakParameters(
            params, 0, cov
        )
        assert almostequal(np.abs(areaDerivs[i]), result.AreaStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(xmaxDerivs[i]), result.PositionStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(ymaxDerivs[i]), result.HeightStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(fwhmDerivs[i]), result.FWHMStdDev, 1e-13, 1e-7)

        assert almostequal(pahf[0], result.Position, 1e-13, 1e-10)
        assert almostequal(pahf[1], result.Area, 1e-13, 1e-10)
        assert almostequal(pahf[2], result.Height, 1e-13, 1e-10)
        assert almostequal(pahf[3], result.FWHM, 1e-13, 1e-10)


def test_derivatives_SecondaryParameters_ExactlyLorentzianLimit():
    # see VoigtArea-Derivatives-ParametrizationSqrtNuLog4.nb
    area = 17
    pos = 7
    w = 3
    nu = 0

    # position, area, height and FWHM
    pahf = [pos, area, 1.80375602170814713871402, 6]
    # derivatives
    ymaxDerivs = [
        0.106103295394596890512589,
        0,
        -0.601252007236049046238005,
        0.502621087962172486855736,
    ]
    xmaxDerivs = [0, 1, 0, 0]
    areaDerivs = [1, 0, 0, 0]
    fwhmDerivs = [0, 0, 2, 0.444686854097446089796046]

    params = lmfit.Parameters()
    params.add("area0", area)
    params.add("pos0", pos)
    params.add("w0", w)
    params.add("nu0", nu)

    for i in range(4):
        cov = np.zeros((4, 4))
        cov[i, i] = 1
        result = VoigtAreaParametrizationNu.GetPositionAreaHeightFWHMFromSinglePeakParameters(
            params, 0, cov
        )
        assert almostequal(np.abs(areaDerivs[i]), result.AreaStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(xmaxDerivs[i]), result.PositionStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(ymaxDerivs[i]), result.HeightStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(fwhmDerivs[i]), result.FWHMStdDev, 1e-13, 1e-7)

        assert almostequal(pahf[0], result.Position, 1e-13, 1e-10)
        assert almostequal(pahf[1], result.Area, 1e-13, 1e-10)
        assert almostequal(pahf[2], result.Height, 1e-13, 1e-10)
        assert almostequal(pahf[3], result.FWHM, 1e-13, 1e-10)


def test_derivatives_SecondaryParameters_NearlyLorentzianLimit():
    # see VoigtArea-Derivatives-ParametrizationSqrtNuLog4.nb
    area = 17
    pos = 7
    w = 3
    nu = 1 / 32767.0

    # position, area, height and FWHM
    pahf = [pos, area, 1.80377136162144979963231, 6.00001501741722156713951]
    # derivatives
    ymaxDerivs = [
        0.106104197742438223507783,
        0,
        -0.601257120540483266544103,
        0.502664788477749705785319,
    ]
    xmaxDerivs = [0, 1, 0, 0]
    areaDerivs = [1, 0, 0, 0]
    fwhmDerivs = [0, 0, 2.00000452341909985950189, 0.444626388979414132950005]

    params = lmfit.Parameters()
    params.add("area0", area)
    params.add("pos0", pos)
    params.add("w0", w)
    params.add("nu0", nu)

    for i in range(4):
        cov = np.zeros((4, 4))
        cov[i, i] = 1
        result = VoigtAreaParametrizationNu.GetPositionAreaHeightFWHMFromSinglePeakParameters(
            params, 0, cov
        )
        assert almostequal(np.abs(areaDerivs[i]), result.AreaStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(xmaxDerivs[i]), result.PositionStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(ymaxDerivs[i]), result.HeightStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(fwhmDerivs[i]), result.FWHMStdDev, 1e-13, 1e-7)

        assert almostequal(pahf[0], result.Position, 1e-13, 1e-10)
        assert almostequal(pahf[1], result.Area, 1e-13, 1e-10)
        assert almostequal(pahf[2], result.Height, 1e-13, 1e-10)
        assert almostequal(pahf[3], result.FWHM, 1e-13, 1e-10)


def test_derivatives_SecondaryParameters_GeneralCase():
    # see VoigtArea-Derivatives-ParametrizationSqrtNuLog4.nb
    area = 17
    pos = 7
    w = 3
    nu = 5 / 7.0

    # position, area, height and FWHM
    pahf = [pos, area, 2.35429674800053710973966, 6.04835222264692750740263]
    # derivatives
    ymaxDerivs = [
        0.138488044000031594690569,
        0,
        -0.784765582666845703246555,
        0.986294916351210818615044,
    ]
    xmaxDerivs = [0, 1, 0, 0]
    areaDerivs = [1, 0, 0, 0]
    fwhmDerivs = [0, 0, 2.01653911742859190892489, -0.134690859032882012668096]

    params = lmfit.Parameters()
    params.add("area0", area)
    params.add("pos0", pos)
    params.add("w0", w)
    params.add("nu0", nu)

    for i in range(4):
        cov = np.zeros((4, 4))
        cov[i, i] = 1
        result = VoigtAreaParametrizationNu.GetPositionAreaHeightFWHMFromSinglePeakParameters(
            params, 0, cov
        )
        assert almostequal(np.abs(areaDerivs[i]), result.AreaStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(xmaxDerivs[i]), result.PositionStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(ymaxDerivs[i]), result.HeightStdDev, 1e-13, 1e-7)
        assert almostequal(np.abs(fwhmDerivs[i]), result.FWHMStdDev, 1e-13, 1e-7)

        assert almostequal(pahf[0], result.Position, 1e-13, 1e-10)
        assert almostequal(pahf[1], result.Area, 1e-13, 1e-10)
        assert almostequal(pahf[2], result.Height, 1e-13, 1e-10)
        assert almostequal(pahf[3], result.FWHM, 1e-13, 1e-10)


def test_parameter_boundaries():
    # add directly, including max
    a = lmfit.Parameters()
    a.add("foo", value=0, max=1)
    assert a["foo"].max == 1

    # add parameter, and then modify
    b = lmfit.Parameters()
    b.add("foo", 1)
    b["foo"].max = 66
    assert b["foo"].max == 66

    # add parameter, and then modify
    c = lmfit.Parameters()
    d = lmfit.Parameter("foo", 34)
    d.max = 77
    c.add(d)
    assert c["foo"] == 34
    assert c["foo"].max == 77
