#!/usr/bin/env python3

import numpy as np
from ramanchada2.fitting_functions.voigtareaparametrizationnu import (
    PositionAreaHeightFWHM,
)
from scipy.special import betaln, gammaln, loggamma

Log2 = 0.69314718055994530941723212145818  # Math.Log(2)


def SafeSqrt(x):
    if x > 0:
        return np.sqrt(x)
    else:
        return 0.0


class PearsonIVAmplitudeParametrizationHPW:

    numberOfTerms = 1
    orderOfBaselinePolynomial = -1

    def __init__(self, numberOfTerms=1, orderOfBackgroundPolynomial=-1):
        self.numberOfTerms = numberOfTerms
        self.orderOfBaselinePolynomial = orderOfBackgroundPolynomial

    def WithNumberOfTerms(self, numberOfTerms):
        return PearsonIVAmplitudeParametrizationHPW(
            numberOfTerms, self.orderOfBaselinePolynomial
        )

    def GetNumberOfParametersPerPeak(self):
        return 5

    def GetYOfOneTerm(x, height, pos, w, m, v):
        arg = np.sqrt((np.power(2, 1 / m) - 1) * (1 + v * v)) * (x - pos) / w - v
        return height * np.exp(
            m
            * (
                np.log((1 + v * v) / (1 + arg * arg))
                - 2 * v * (np.arctan(arg) + np.arctan(v))
            )
        )

    def func(self, pars, x, data=None):
        sum = np.zeros(len(x))
        for i in range(self.orderOfBaselinePolynomial, -1, -1):
            sum *= x
            sum += pars[f"b{i}"]

        for i in range(self.numberOfTerms):
            a, xc, w, m, v = (
                pars[f"a{i}"],
                pars[f"pos{i}"],
                pars[f"w{i}"],
                pars[f"m{i}"],
                pars[f"v{i}"],
            )
            arg = np.sqrt((np.power(2, 1 / m) - 1) * (1 + v * v)) * (x - xc) / w - v
            sum += a * np.exp(
                m
                * (
                    np.log((1 + v * v) / (1 + arg * arg))
                    - 2 * v * (np.arctan(arg) + np.arctan(v))
                )
            )

        if data is None:
            return sum
        return sum - data

    def dfunc(self, pars, x, data=None):
        result = []
        k = 0

        # first the derivatives of the peak terms
        for i in range(self.numberOfTerms):
            height, pos, w, m, v = (
                pars[f"a{i}"],
                pars[f"pos{i}"],
                pars[f"w{i}"],
                pars[f"m{i}"],
                pars[f"v{i}"],
            )
            twoToOneByM_1 = np.power(2, 1 / m) - 1
            ww = w / np.sqrt(twoToOneByM_1 * (1 + v * v))
            z = (x - pos) / ww - v
            log1v2_1z2 = np.log((1 + v * v) / (1 + z * z))
            atanZ_p_V = 2 * v * (np.arctan(z) + np.arctan(v))
            body = np.exp(m * (log1v2_1z2 - atanZ_p_V))
            dbodydz = -body * (2 * m * (z + v)) / (1 + z * z)

            dfdheight = np.full(len(x), np.NaN)
            dfdpos = np.full(len(x), np.NaN)
            dfdw = np.full(len(x), np.NaN)
            dfdm = np.full(len(x), np.NaN)
            dfdv = np.full(len(x), np.NaN)

            dfdheight = body
            dfdpos = -height * dbodydz / ww
            dfdw = -height * dbodydz * (z + v) / w
            dfdm = (
                height
                * body
                * (
                    -atanZ_p_V
                    + log1v2_1z2
                    + Log2
                    * np.square(z + v)
                    * np.power(2, 1 / m)
                    / (twoToOneByM_1 * (1 + z * z) * m)
                )
            )
            dfdv = (
                -height
                * 2
                * m
                * body
                * (
                    (z + v) * (z * v - 1) / ((1 + z * z) * (1 + v * v))
                    + np.arctan(z)
                    + np.arctan(v)
                )
            )

            result.append(dfdheight)
            result.append(dfdpos)
            result.append(dfdw)
            result.append(dfdm)
            result.append(dfdv)
            k += 5

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
        if not (relativeHeight > 0 and relativeHeight < 1):
            raise ValueError("RelativeHeight should be in the open interval (0, 1)")

        # we evaluate the parameters here for a pure Lorentzian (m=1, v=0)
        w = np.abs(0.5 * fullWidth * np.sqrt(relativeHeight / (1 - relativeHeight)))
        return [height, position, w, 1, 0]  # Parameters for the Lorentz limit

    def AddInitialParametersFromHeightPositionAndWidthAtRelativeHeight(
        paras, indexOfPeak, height, position, fullWidth, relativeHeight=0.5
    ):
        apwmv = PearsonIVAmplitudeParametrizationHPW.GetInitialParametersFromHeightPositionAndWidthAtRelativeHeight(
            height, position, fullWidth, relativeHeight
        )
        paras.add(f"area{indexOfPeak}", apwmv[0])
        paras.add(f"pos{indexOfPeak}", apwmv[1])
        paras.add(f"w{indexOfPeak}", apwmv[2])
        paras.add(f"m{indexOfPeak}", apwmv[3])
        paras.add(f"v{indexOfPeak}", apwmv[3])

    def GetFWHMApproximation(w, m, v):
        sq = np.sqrt((np.power(2, 1 / m) - 1) * (1 + v * v))
        ws = w / sq
        ms = m
        vs = 2 * m * v
        return (
            ws
            * np.sqrt(np.power(2, 1 / ms) - 1)
            * (np.pi / np.arctan2(np.exp(1) * ms, np.abs(vs)))
        )

    def GetFWHM(w, m, v):
        return PearsonIVAmplitudeParametrizationHPW.GetHWHM(
            w, m, v, True
        ) + PearsonIVAmplitudeParametrizationHPW.GetHWHM(w, m, v, False)

    def GetHWHM(w, m, v, rightSide):
        sq = np.sqrt((np.power(2, 1 / m) - 1) * (1 + v * v))
        return PearsonIVAmplitudeParametrizationHPW.GetHWHMOfPearson4Original(
            w / sq, m, 2 * m * v, rightSide
        )

    def GetHWHMOfPearson4Original(w, m, v, rightSide):
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

    def GetArea(amp, w, m, v):
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

    def GetPositionAreaHeightFWHMFromSinglePeakParameters(
        self, parameters, indexOfPeak, cv=None
    ):
        amp = parameters[f"a{indexOfPeak}"]
        loc = parameters[f"pos{indexOfPeak}"]
        wm = parameters[f"w{indexOfPeak}"]
        m = parameters[f"m{indexOfPeak}"]
        vm = parameters[f"v{indexOfPeak}"]

        area = PearsonIVAmplitudeParametrizationHPW.GetArea(amp, wm, m, vm)
        pos = loc
        height = amp
        fwhm = PearsonIVAmplitudeParametrizationHPW.GetHWHM(
            wm, m, vm, True
        ) + PearsonIVAmplitudeParametrizationHPW.GetHWHM(wm, m, vm, False)

        posStdDev = 0
        areaStdDev = 0
        heightStdDev = 0
        fwhmStdDev = 0

        if cv is not None:
            deriv = np.zeros(5)
            resVec = np.zeros(5)

            # PositionVariance
            posStdDev = SafeSqrt(cv[1, 1])

            # Height variance
            heightStdDev = SafeSqrt(cv[0, 0])

            # Area variance
            deriv[0] = PearsonIVAmplitudeParametrizationHPW.GetArea(1, wm, m, vm)
            deriv[1] = 0
            absDelta = wm * 1e-5
            deriv[2] = (
                PearsonIVAmplitudeParametrizationHPW.GetArea(amp, wm + absDelta, m, vm)
                - PearsonIVAmplitudeParametrizationHPW.GetArea(
                    amp, wm - absDelta, m, vm
                )
            ) / (2 * absDelta)
            absDelta = m * 1e-5
            deriv[3] = (
                PearsonIVAmplitudeParametrizationHPW.GetArea(amp, wm, m + absDelta, vm)
                - PearsonIVAmplitudeParametrizationHPW.GetArea(
                    amp, wm, m - absDelta, vm
                )
            ) / (2 * absDelta)
            absDelta = 1e-5 if vm == 0 else np.abs(vm * 1e-5)
            deriv[4] = (
                PearsonIVAmplitudeParametrizationHPW.GetArea(amp, wm, m, vm + absDelta)
                - PearsonIVAmplitudeParametrizationHPW.GetArea(
                    amp, wm, m, vm - absDelta
                )
            ) / (2 * absDelta)
            resVec = cv.dot(deriv)
            areaStdDev = SafeSqrt(np.dot(deriv, resVec))

            # FWHM variance
            deriv[0] = 0
            deriv[1] = 0
            absDelta = wm * 1e-5
            deriv[2] = (
                PearsonIVAmplitudeParametrizationHPW.GetFWHM(wm + absDelta, m, vm)
                - PearsonIVAmplitudeParametrizationHPW.GetFWHM(wm - absDelta, m, vm)
            ) / (2 * absDelta)
            absDelta = m * 1e-5
            deriv[3] = (
                PearsonIVAmplitudeParametrizationHPW.GetFWHM(wm, m + absDelta, vm)
                - PearsonIVAmplitudeParametrizationHPW.GetFWHM(wm, m - absDelta, vm)
            ) / (2 * absDelta)
            absDelta = 1e-5 if vm == 0 else np.abs(vm * 1e-5)
            deriv[4] = (
                PearsonIVAmplitudeParametrizationHPW.GetFWHM(wm, m, vm + absDelta)
                - PearsonIVAmplitudeParametrizationHPW.GetFWHM(wm, m, vm - absDelta)
            ) / (2 * absDelta)
            resVec = cv.dot(deriv)
            fwhmStdDev = SafeSqrt(np.dot(deriv, resVec))

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
        DefaultMinWidth = 1.4908919308538355e-81  # Math.Pow(double.Epsilon, 0.25);
        DefaultMaxWidth = 1.157920892373162e77  # Math.Pow(double.MaxValue, 0.25);

        for i in range(self.numberOfTerms):
            params[f"a{i}"].min = 0  # minimal amplitude is 0
            if minimalPosition is not None:
                params[f"pos{i}"].min = minimalPosition
            if maximalPosition is not None:
                params[f"pos{i}"].max = maximalPosition
            if minimalFWHM is not None:
                params[f"w{i}"].min = minimalFWHM / 2.0
            else:
                params[f"w{i}"].min = DefaultMinWidth
            if maximalFWHM is not None:
                params[f"w{i}"].max = maximalFWHM / 2.0
            else:
                params[f"w{i}"].max = DefaultMaxWidth
            params[f"m{i}"].min = 1 / 2.0 + 1 / 1024.0
            params[f"m{i}"].max = 1000
            params[f"v{i}"].min = -1000
            params[f"v{i}"].max = 1000
