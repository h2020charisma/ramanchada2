import numpy as np
from scipy.special import erfc, voigt_profile

DBL_EPSILON = np.finfo(float).eps
OneBySqrtLog4 = 0.84932180028801904272150283410289  # 1/sqrt(log(4))
Log4 = 1.3862943611198906188344642429164  # Log(4)
Sqrt2 = 1.4142135623730950488016887242097  # Math.Sqrt(2)
Sqrt2Pi = 2.5066282746310005024157652848110  # Math.Sqrt(2*Math.Pi)
SqrtLog2 = 0.832554611157697756353165  # Math.Sqrt(Math.Log(2))
Sqrt2PiByLog4 = 2.12893403886245235863054  # Math.Sqrt(2*Math.Pi/Math.Log(4))
SqrtLog4 = 1.1774100225154746910  # Math.Sqrt(Math.Log(4))
SqrtPi = 1.77245385090551602729817  # Math.Sqrt(Math.PI)
Log2 = 0.693147180559945309417232  # Math.Log(2)
Log2P32 = 0.577082881386139784459964  # Math.Pow(Log2, 1.5)
SqrtPiLog2 = 1.47566462663560588938882  # Math.Sqrt(Math.PI*Math.Log(2))
C2_FWHM = 0.21669  # Approximation constant for FWHM of Voigt
C1_FWHM = 1 - np.sqrt(C2_FWHM)  # Second constant for FWHM of Voigt


def voigt_nu_profile(x, amplitude=1.0, center=0.0, sigma=1.0, nu=1.0):
    """Returns the y-value of one peak in dependence on x and the peak parameters."""
    return amplitude * voigt_profile(
        x - center, sigma * np.sqrt(nu) * OneBySqrtLog4, sigma * (1 - nu)
    )


def voigt_nu_FWHM_approx(*, w, nu):
    sigma = w * np.sqrt(nu) * OneBySqrtLog4
    gamma = w * (1 - nu)
    return 2*voigt_nu_HWHM_approx(sigma=sigma, gamma=gamma)


def voigt_nu_FWHM(*, w, nu):
    sigma = w * np.sqrt(nu) * OneBySqrtLog4
    gamma = w * (1 - nu)
    return 2*voigt_nu_HWHM(sigma=sigma, gamma=gamma)


def voigt_nu_HWHM_approx(*, sigma, gamma):
    """
    Gets the approximate half-width-half-maximum of a peak
    in dependence on the original Voigt function parameters 'sigma' and 'gamma'.
    The approximation has an accuracy of 0.0216%.
    """
    C2 = 0.86676  # for relative error always < 0.000216 _and_ asymptotic correct behaviour
    C2S = 2 - np.sqrt(C2)
    return 0.5 * (
        C2S * gamma
        + np.sqrt(C2 * gamma * gamma + 4 * 2 * np.log(2) * sigma * sigma)
    )


def voigt_nu_HWHM(*, sigma, gamma):
    """
    Gets the half-width-half-maximum of a peak
    in dependence on the original Voigt function parameters 'sigma' and 'gamma'.
    """
    side = 0
    if sigma == 0 and gamma == 0:
        return 0
    if np.isnan(sigma) or np.isnan(gamma):
        return np.NaN

    # Reduce order of magnitude to prevent overflow
    prefac = 1.0
    s = np.abs(sigma)
    g = np.abs(gamma)
    while s > 1e100 or g > 1e100:
        prefac *= 1e30
        s /= 1e30
        g /= 1e30

    # Increase order of magnitude to prevent underflow
    while s < 1e-100 and g < 1e-100:
        prefac /= 1e30
        s *= 1e30
        g *= 1e30

    HM = voigt_profile(0.0, s, g) / 2

    # Choose initial points a,b that bracket the expected root
    c = voigt_nu_HWHM_approx(sigma=s, gamma=g)
    a = c * 0.995
    b = c * 1.005
    del_a = voigt_profile(a, s, g) - HM
    del_b = voigt_profile(b, s, g) - HM

    # Iteration using regula falsi (Illinois variant).
    # Empirically, this takes <5 iterations to converge to FLT_EPSILON
    # and <10 iterations to converge to DBL_EPSILON.
    # We have never seen convergence worse than k = 15.
    for k in range(30):
        if np.abs(del_a - del_b) < 2 * DBL_EPSILON * HM:
            return prefac * (a + b) / 2
        c = (b * del_a - a * del_b) / (del_a - del_b)
        if np.abs(b - a) < 2 * DBL_EPSILON * np.abs(b + a):
            return prefac * c
        del_c = voigt_profile(c, s, g) - HM

        if del_b * del_c > 0:
            b = c
            del_b = del_c
            if side < 0:
                del_a /= 2
            side = -1
        elif del_a * del_c > 0:
            a = c
            del_a = del_c
            if side > 0:
                del_b /= 2
            side = 1
        else:
            return prefac * c


def voigt_nu_height(*, amplitude, sigma, nu):
    area = amplitude
    w = sigma

    sigma = w * np.sqrt(nu) * OneBySqrtLog4
    gamma = w * (1 - nu)

    if nu < (1 / 20.0):  # approximation by series expansion is needed in the Lorentz limit
        expErfcTermBySqrtNu = (
            (
                (
                    ((4.21492418972454262863068) * nu + 0.781454843465470036843478)
                    * nu
                    + 0.269020594012510850443784
                )
                * nu
                + 0.188831848731661377618477
            )
            * nu
        ) + 0.677660751603104996618259
    else:  # normal case
        expTerm = np.exp(0.5 * np.square(gamma / sigma))
        erfcTerm = erfc(-gamma / (np.sqrt(2) * sigma))
        expErfcTermBySqrtNu = expTerm * (2 - erfcTerm) / np.sqrt(nu)

    bodyheight = expErfcTermBySqrtNu / (w * OneBySqrtLog4 * Sqrt2Pi)
    height = area * bodyheight
    return height
