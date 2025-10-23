import numpy as np
from scipy.special import betaln, gammaln, loggamma


def perarson4_hpw_profile(x, height=1.0, center=0.0, sigma=1.0, expon=1.0, skew=0.0):
    """Returns the y-value of one peak in dependence on x and the peak parameters."""
    np.sqrt((2**(1/expon)-1) * (1+skew**2)) * (x-center) / sigma - skew

    arg = (
        np.sqrt((np.power(2, 1 / expon) - 1) * (1 + skew * skew))
        * (x - center)
        / sigma
        - skew
    )
    return height * np.exp(
        expon
        * (
            np.log((1 + skew * skew) / (1 + arg * arg))
            - 2 * skew * (np.arctan(arg) + np.arctan(skew))
        )
    )


def pearson4_hpw_area(amp, w, m, v):
    lnprefactor = (
        betaln(m - 0.5, 0.5)
        + 2 * gammaln(m)
        - 2 * np.real(loggamma(m + m * v * 1.0j))
    )
    tmm1 = np.power(2, 1 / m) - 1
    lnbody = (
        -2 * m * v * np.arctan(v)
        + (m - 0.5) * np.log(1 + v * v)
        - 0.5 * np.log(tmm1)
    )
    return amp * w * np.exp(lnprefactor + lnbody)


def pearson4_hpw_FWHM_approx(w, m, v):
    """
    Gets the approximate full-width-half-maximum of a peak
    in dependence on the parameters 'w', 'm' and 'v'.
    The approximation has an accuracy of 39%.
    """
    sq = np.sqrt((np.power(2, 1 / m) - 1) * (1 + v * v))
    ws = w / sq
    ms = m
    vs = 2 * m * v
    return (
        ws
        * np.sqrt(np.power(2, 1 / ms) - 1)
        * (np.pi / np.arctan2(np.exp(1) * ms, np.abs(vs)))
    )


def pearson4_hpw_LeftWHM(w, m, v):
    sq = np.sqrt((np.power(2, 1 / m) - 1) * (1 + v * v))
    return _pearson4_original_HFHM(w / sq, m, 2 * m * v, False)


def pearson4_hpw_RightWHM(w, m, v):
    sq = np.sqrt((np.power(2, 1 / m) - 1) * (1 + v * v))
    return _pearson4_original_HFHM(w / sq, m, 2 * m * v, True)


def pearson4_hpw_FWHM(w, m, v):
    """
    Gets the full-width-half-maximum of a peak
    in dependence on the parameters 'w', 'm' and 'v'.
    """
    return pearson4_hpw_LeftWHM(w, m, v) + pearson4_hpw_RightWHM(w, m, v)


def _pearson4_original_HFHM(w, m, v, rightSide):
    """
    Gets the half-width-half-maximum of the side of a peak
    in dependence on the parameters 'w', 'm' and 'v'.
    ATTENTION: Here, the parameters 'w', 'm' and 'v'
    are from the original definition of PearsonIV, not
    the parametrized definition used in this class!
    If 'rightSide' is true, the HWHM of the right side
    of the peak is returned, else that of the left side.
    """
    if not m > 0:
        return np.NaN

    w = np.abs(w)
    sign = 1 if rightSide else -1
    z0 = -v / (2 * m)

    def funcsimp(z, m, v):
        return (
            np.log(2)
            + v * (np.arctan(z0) - np.arctan(z))
            + m
            * (
                np.log(1 + z0 * z0)
                - (
                    2 * np.log(np.abs(z))
                    if np.abs(z) > 1e100
                    else np.log(1 + z * z)
                )
            )
        )

    def dervsimp(z, m, v):
        return (
            (-v - 2 * m * z) / (1 + z * z)
            if np.abs(z) < 1e100
            else (-v / z - 2 * m) / z
        )

    # go forward in exponentially increasing steps, until the amplitude falls below ymaxHalf,
    # in order to bracked the solution
    zNear = z0
    zFar = z0
    d = 1.0

    while np.isfinite(d):
        zFar = z0 + d * sign
        y = funcsimp(zFar, m, v)
        if y < 0:
            break
        else:
            zNear = zFar
        d *= 2
    if zNear > zFar:
        (zNear, zFar) = (zFar, zNear)

    # use Newton-Raphson to refine the result
    z = 0.5 * (zNear + zFar)  # starting value
    for i in range(40):
        funcVal = funcsimp(z, m, v)
        if rightSide:
            if funcVal > 0 and z > zNear:
                zNear = z
            if funcVal < 0 and z < zFar:
                zFar = z
        else:  # leftSide
            if funcVal < 0 and z > zNear:
                zNear = z
            if funcVal > 0 and z < zFar:
                zFar = z

        dz = funcVal / dervsimp(z, m, v)
        znext = z - dz
        if znext <= zNear:
            znext = (z + zNear) / 2
        elif znext >= zFar:
            znext = (z + zFar) / 2

        if z == znext:
            break

        z = znext

        if np.abs(dz) < 5e-15 * np.abs(z):
            break
    return np.abs((z - z0) * w)
