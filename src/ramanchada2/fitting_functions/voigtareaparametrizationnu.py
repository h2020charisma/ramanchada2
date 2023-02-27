#!/usr/bin/env python3

from decimal import InvalidOperation
import numpy as np
from scipy.special import voigt_profile, wofz, erfc
from collections import namedtuple


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


def SafeSqrt(x):
    """Returns 0 if x is negative, otherwise the square root of x"""
    return 0.0 if x < 0 else np.sqrt(x)


# Gets the maximal relative error of the VoigtHalfWidthHalfMaximumApproximation(sigma, gamma) function
VoigtHalfWidthHalfMaximumApproximationMaximalRelativeError = 0.000216

PositionAreaHeightFWHM = namedtuple(
    "PositionAreaHeightFWHM",
    [
        "Position",
        "PositionStdDev",
        "Area",
        "AreaStdDev",
        "Height",
        "HeightStdDev",
        "FWHM",
        "FWHMStdDev",
    ],
)


class VoigtAreaParametrizationNu:
    """
    Voigt peak fitting function with multiple peaks and polynomial baseline

    A special parametrization is used for the Voigt function:
    Each of the peaks has 4 parameters: area, pos, w, nu
    Parameter 'area' designates the area under the peak
    Parameter 'pos' designates the position of the peak (x-value at maximum)
    Parameter 'w' is approximately (2%) the HWHM value of the peak
    Parameter 'nu' (0..1) determines the Gauss'ness of the peak (0=Lorentzian,
                1=Gaussian)

    Note: the parameters 'sigma' and 'gamma' of the original Voigt function are
          calculated as follows:
          sigma = w * Sqrt(nu / log(4))
          gamma = w * (1 - nu)
    """

    numberOfTerms = 1
    orderOfBaselinePolynomial = -1

    def __init__(self, numberOfTerms=1, orderOfBackgroundPolynomial=-1):
        """
        Initializes an instance of the class with a given number of
        peak terms, and the order of the baseline polynomial
        """
        self.numberOfTerms = numberOfTerms
        self.orderOfBaselinePolynomial = orderOfBackgroundPolynomial

    def WithNumberOfTerms(self, numberOfTerms):
        """Returns a new instance of the class with the given number of peak terms."""
        return VoigtAreaParametrizationNu(numberOfTerms, self.orderOfBaselinePolynomial)

    def GetNumberOfParametersPerPeak(self):
        """Returns the number of parameters per peak term."""
        return 4

    def func(self, pars, x, data=None):
        """Returns the y-values of the fitting function."""
        sum = np.zeros(len(x))
        for i in range(self.orderOfBaselinePolynomial, -1, -1):
            sum *= x
            sum += pars[f"b{i}"]

        for i in range(self.numberOfTerms):
            a, xc, w, nu = (
                pars[f"area{i}"],
                pars[f"pos{i}"],
                pars[f"w{i}"],
                pars[f"nu{i}"],
            )
            sum += a * voigt_profile(
                x - xc, w * np.sqrt(nu) * OneBySqrtLog4, w * (1 - nu)
            )

        if data is None:
            return sum
        return sum - data

    def dfunc(self, pars, x, data=None):
        """Returns the derivatives of the fitting function w.r.t. the parameters."""
        result = []
        k = 0

        # first the derivatives of the peak terms
        for i in range(self.numberOfTerms):
            area, xc, w, nu = (
                pars[f"area{i}"],
                pars[f"pos{i}"],
                pars[f"w{i}"],
                pars[f"nu{i}"],
            )
            arg = x - xc
            sigma = w * np.sqrt(nu) * OneBySqrtLog4
            gamma = w * (1 - nu)

            dfdarea = np.full(len(x), np.NaN)
            dfdpos = np.full(len(x), np.NaN)
            dfdw = np.full(len(x), np.NaN)
            dfdnu = np.full(len(x), np.NaN)

            if w > 0 and nu >= 0:
                if nu < 1e-4:  # approximately this is a Lorentzian
                    arg /= w
                    onePlusArg2 = 1 + arg * arg
                    body = 1 / (np.pi * w * onePlusArg2)
                    dfdarea = body
                    dfdpos = area * body * 2 * arg / (w * onePlusArg2)
                    dfdw = -area * body * (1 - arg * arg) / (w * onePlusArg2)
                    dfdnu = (
                        area
                        * body
                        * (arg * arg * (3 - arg * arg * Log4) + Log4 - 1)
                        / (onePlusArg2 * onePlusArg2 * Log4)
                    )
                elif (
                    nu <= 1
                ):  # general case including nu==1 (which means gamma==0, i.e. pure Gaussian)
                    arg_gammaj = np.empty(len(arg), dtype=np.complex128)
                    arg_gammaj.real = arg
                    arg_gammaj.imag = gamma  # arg_gammaj is now complex(arg, gamma)
                    z = arg_gammaj / (Sqrt2 * sigma)
                    wOfZ = wofz(z)
                    body = wOfZ / (Sqrt2Pi * sigma)
                    dbodydz = (Sqrt2 * 1.0j - z * wOfZ * Sqrt2Pi) / (
                        np.pi * sigma
                    )  # Derivative of wOfZBySqrt2PiSigma w.r.t. z
                    dfdarea = np.real(body)  # Derivative w.r.t. amplitude
                    dfdpos = (
                        -area * np.real(dbodydz) / (Sqrt2 * sigma)
                    )  # Derivative w.r.t. position
                    dfdw = (
                        -area / w * np.real(dbodydz * arg / (Sqrt2 * sigma) + body)
                    )  # Derivative w.r.t. w
                    argwonepnu = np.empty(len(arg), dtype=np.complex128)
                    argwonepnu.real = arg
                    argwonepnu.imag = w * (1 + nu)  # complex(arg, w * (1 + nu))
                    dfdnu = (
                        -area
                        / (2 * nu)
                        * np.real((dbodydz * argwonepnu / (Sqrt2 * sigma) + body))
                    )  # Derivative w.r.t. nu

            result.append(dfdarea)
            result.append(dfdpos)
            result.append(dfdw)
            result.append(dfdnu)
            k += 4

        # second the baseline derivatives
        if self.orderOfBaselinePolynomial >= 0:
            xn = np.ones(len(x))
            for i in range(self.orderOfBaselinePolynomial + 1):
                result.append(np.copy(xn))
                k += 1
                xn *= x
        return np.array(result)

    def GetInitialParametersFromHeightPositionAndWidthAtRelativeHeight(
        height, position, fullWidth, relativeHeight=0.5
    ):
        """
        Returns an array of initial parameters for peak fitting,
        given the height, the position, the full width and the relative
        height at which the full width was measured.
        """
        useLorenzLimit = False

        if not (relativeHeight > 0 and relativeHeight < 1):
            raise ValueError("RelativeHeight should be in the open interval (0, 1)")

        if useLorenzLimit:
            # we calculate at the Lorentz limit (nu==0)
            w = 0.5 * fullWidth * np.sqrt(relativeHeight / (1 - relativeHeight))
            area = height * w * np.pi
            return [area, position, w, 0]
        else:
            # use Gaussian limit
            w = 0.5 * fullWidth * SqrtLog2 / np.sqrt(-np.log(relativeHeight))
            area = height * w * Sqrt2PiByLog4
            return [area, position, w, 1]

    def AddInitialParametersFromHeightPositionAndWidthAtRelativeHeight(
        paras, indexOfPeak, height, position, fullWidth, relativeHeight=0.5
    ):
        """
        Add to an existing 'Parameters' instance the initial parameters for a peak.

        Parameters
        ----------
        paras: Parameters
            Existing instance of Parameters, to which the new parameters will be added.
        indexOfPeak:
            The index of the peak whose parameters are added.
        height:
            Approximate height of the peak.
        pos:
            Approximate position of the peak
        fullWidth:
            Approximate full width of the peak
        relativeHeight:
            Relative height, at which full width was measured (usually 0.5)
        """
        apwnu = VoigtAreaParametrizationNu.GetInitialParametersFromHeightPositionAndWidthAtRelativeHeight(
            height, position, fullWidth, relativeHeight
        )
        paras.add(f"area{indexOfPeak}", apwnu[0])
        paras.add(f"pos{indexOfPeak}", apwnu[1])
        paras.add(f"w{indexOfPeak}", apwnu[2])
        paras.add(f"nu{indexOfPeak}", apwnu[3])

    def VoigtHalfWidthHalfMaximumOfSigmaGammaApproximation(sigma, gamma):
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

    def VoigtHalfWidthHalfMaximumOfWNuApproximation(w, nu):
        """
        Gets the approximate half-width-half-maximum of a peak
        in dependence on the parameters 'w' and 'nu'.
        The approximation has an accuracy of 0.0216%.
        """
        sigma = w * np.sqrt(nu) * OneBySqrtLog4
        gamma = w * (1 - nu)
        return VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGammaApproximation(
            sigma, gamma
        )

    def VoigtHalfWidthHalfMaximumOfSigmaGamma(sigma, gamma):
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
        c = VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGammaApproximation(
            s, g
        )
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

        raise InvalidOperation  # One should never arrive here

    def VoigtHalfWidthHalfMaximumOfWNu(w, nu):
        """
        Gets the half-width-half-maximum of a peak
        in dependence on the parameters 'w' and 'nu'.
        """
        sigma = w * np.sqrt(nu) * OneBySqrtLog4
        gamma = w * (1 - nu)
        return VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGamma(
            sigma, gamma
        )

    def GetPositionAreaHeightFWHMFromSinglePeakParameters(
        self, parameters, indexOfPeak, cv=None
    ):
        """
        Get position, area, height, and FWHM of one peak from the peak parameters.
        If the covariance matrix is given, the corresponding errors are calculated.

        Parameters
        ----------
        parameters: Parameters
            Existing instance of Parameters, which contains the result of the fit.
        indexOfPeak: int
            The index of the peak into consideration.
        cv: np.array
            Covariance matrix of the fit.
        """
        area = parameters[f"area{indexOfPeak}"]
        pos = parameters[f"pos{indexOfPeak}"]
        w = parameters[f"w{indexOfPeak}"]
        nu = parameters[f"nu{indexOfPeak}"]
        sigma = w * np.sqrt(nu) * OneBySqrtLog4
        gamma = w * (1 - nu)

        if cv is not None:
            cv = cv[
                indexOfPeak * 4: (indexOfPeak + 1) * 4,
                indexOfPeak * 4: (indexOfPeak + 1) * 4,
            ]

        if cv is not None:
            areaStdDev = SafeSqrt(cv[0, 0])
            posStdDev = SafeSqrt(cv[1, 1])
        else:
            areaStdDev = 0.0
            posStdDev = 0.0

        height = np.NaN
        fwhm = np.NaN
        heightStdDev = 0.0
        fwhmStdDev = 0.0

        # expErfcTerm is normally: Math.Exp(0.5 * RMath.Pow2(gamma / sigma)) * Erfc(gamma / (Math.Sqrt(2) * sigma))
        # but for large gamma/sigma, we need a approximation, because the exp term becomes too large
        expErfcTermBySqrtNu = np.NaN

        # for gamma > 20*sigma we need an approximation of the expErfcTerm,
        # since the expTerm will get too large and the Erfc term too small
        # we use a series expansion

        if nu < (
            1 / 20.0
        ):  # approximation by series expansion is needed in the Lorentz limit
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

        # fwhmSqrtTerm = np.sqrt(C2_FWHM * gamma * gamma + 8 * sigma * sigma * np.log(2));
        # fwhm = C1_FWHM * gamma + fwhmSqrtTerm; // this is the approximation formula
        fwhm = 2 * VoigtAreaParametrizationNu.VoigtHalfWidthHalfMaximumOfSigmaGamma(
            sigma, gamma
        )  # we use the exact formula for fwhm

        if cv is not None:
            dHeightByDArea = bodyheight
            dHeightByDW = -bodyheight * area / w
            dHeightByDNu = np.NaN

        if nu < (1 / 20.0):
            # Series expansion for Lorentzian limit
            dHeightByDNu = (
                (
                    (
                        (
                            (
                                (
                                    88758.4100339208444038841 * nu
                                    - 7108.49617246574864439307
                                )
                                * nu
                                + 641.286392761620402559762
                            )
                            * nu
                            - 66.1018151520036002556775
                        )
                        * nu
                        + 7.91931382144031445585169
                    )
                    * nu
                    - 1.10119171735779477287440
                )
                * nu
                + 0.252727974753276908699454
            ) * nu + 0.0886978390521480859157182
        else:
            # General case
            dHeightByDNu = (
                (1 + nu) * Log2
                - (expErfcTermBySqrtNu)
                * ((1 - nu * nu) * SqrtPi * Log2P32 + 0.5 * nu * SqrtPiLog2)
            ) / (nu * nu * np.pi)

        dHeightByDNu *= area / w

        heightStdDev = SafeSqrt(
            cv[0, 0] * np.square(dHeightByDArea)
            + cv[2, 2] * np.square(dHeightByDW)
            + cv[3, 3] * np.square(dHeightByDNu)
            + dHeightByDNu * (cv[0, 3] + cv[3, 0]) * dHeightByDArea
            + dHeightByDW * (cv[0, 2] + cv[2, 0]) * dHeightByDArea
            + dHeightByDNu * (cv[2, 3] + cv[3, 2]) * dHeightByDW
        )

        dFwhmByDW = 2 * (
            1
            + np.sqrt(C2_FWHM) * (-1 + nu)
            - nu
            + np.sqrt(C2_FWHM * np.square(1 - nu) + nu)
        )

        dFwhmByDNu = (
            2
            * w
            * (
                -1
                + np.sqrt(C2_FWHM)
                + (1 + 2 * C2_FWHM * (-1 + nu))
                / (2 * np.sqrt(C2_FWHM * np.square(1 - nu) + nu))
            )
        )

        fwhmStdDev = SafeSqrt(
            cv[3, 3] * np.square(dFwhmByDNu)
            + dFwhmByDW * ((cv[2, 3] + cv[3, 2]) * dFwhmByDNu + cv[2, 2] * dFwhmByDW)
        )

        return PositionAreaHeightFWHM(
            pos, posStdDev, area, areaStdDev, height, heightStdDev, fwhm, fwhmStdDev
        )

    def SetParameterBoundariesForPositivePeaks(
        self,
        params,
        minimalPosition=None,
        maximalPosition=None,
        minimalFWHM=None,
        maximalFWHM=None,
    ):
        """
        Amends the parameters in an existing Parameters instance
        with fit boundaries.

        Parameters
        ----------
        params: Parameters
            Existing instance of Parameters, which already contain all parameters.
        minimalPosition: float (or None)
            The minimal allowed position of a peak.
        maximalPosition: float (or None)
            The maximal allowed position of a peak.
        minimalFWHM: float (or None)
            The minimal allowed FWHM value of a peak.
        maximalFWHM: float (or None)
            The maximal allowed FWHM value of a peak.

        Note: the minimal height is set to 0 in order to disallow negative peaks.
              the maximal height of a peak is not limited
        """
        for i in range(self.numberOfTerms):
            params[f"area{i}"].min = 0  # minimal area is 0
            if minimalPosition is not None:
                params[f"pos{i}"].min = minimalPosition
            if maximalPosition is not None:
                params[f"pos{i}"].max = maximalPosition
            if minimalFWHM is not None:
                params[f"w{i}"].min = minimalFWHM / 2.0
            else:
                params[f"w{i}"].min = np.sqrt(np.sqrt(DBL_EPSILON))
            if maximalFWHM is not None:
                params[f"w{i}"].max = maximalFWHM / 2.0
            params[f"nu{i}"].min = 0
            params[f"nu{i}"].max = 1
