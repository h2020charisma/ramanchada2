import numpy as np
from scipy import signal
import lmfit


def IsInRange(x, idx):
    return idx >= 0 and idx <= len(x) - 1


def InterpolateLinear(index, array, extendToSides=False):
    if not (index >= 0):
        if extendToSides:
            return array[0]
        else:
            raise AttributeError
    elif not (index <= len(array) - 1):
        if extendToSides:
            return array[-1]
        else:
            raise AttributeError
    elif index == len(array) - 1:
        return array[-1]

    idx = np.floor(index)
    fraction = index - idx
    if fraction == 0:
        return array[idx]  # return left value
    elif fraction == 1:
        return array[idx + 1]  # return right value
    else:
        return (1 - fraction) * array[idx] + fraction * array[idx + 1]


def GetWidthValue(x, leftIdx, middleIdx, rightIdx):
    if IsInRange(x, leftIdx) and IsInRange(x, rightIdx):
        return InterpolateLinear(rightIdx, x) - InterpolateLinear(leftIdx, x)
    elif IsInRange(x, middleIdx) and IsInRange(x, leftIdx):
        return 2 * (InterpolateLinear(middleIdx, x) - InterpolateLinear(leftIdx, x))
    elif IsInRange(x, rightIdx) and IsInRange(x, middleIdx):
        return 2 * (InterpolateLinear(rightIdx, x) - InterpolateLinear(middleIdx, x))
    else:
        # Try to interpolate over the full range of x
        return (x[-1] - x[0]) * (rightIdx - leftIdx) / (len(x) - 1.0)


class FindPeaksByIncrementalPeakAddition:

    # Order of the baseline polynomial (-1=no baseline, 0=constant, 1=linear, etc.)
    orderOfBaselinePolynomial = -1

    # Maximum number of peak that will be fitted
    maximumNumberOfPeaks = 50

    minimalRelativeHeight = 2.5e-3

    minimalSignalToNoiseRatio = 8

    fitWidthScalingFactor = 2

    prunePeaksSumChiSquareFactor = 0.1

    minimalFWHMValue = None

    isMinimalFWHMValueInXUnits = True

    def Execute(self, x, y, fitFunc):
        # estimate the properties of the x and y array

        minimalX = np.min(x)
        maximalX = np.max(x)
        minimalY = np.min(y)
        maximalY = np.max(y)
        minIncrement = np.min(np.abs(x[1:] - x))
        spanX = maximalX - minimalX
        spanY = maximalY - minimalY

        # estimate the noise level
        noiseArray = np.abs(y[1:] - 0.5 * y[2:] - 0.5 * y)
        np.sort(noiseArray)
        noiseLevel = (
            noiseArray[len(noiseArray) / 2] * 1.22
        )  # take the 50% percentile ans noise level

        fitFunction = fitFunc.WithNumberOfTerms(1).WithOrderOfBaselinePolynomial(
            self.orderOfBaselinePolynomial
        )

        prohibitedPeaks = {}

        yRest = np.copy(y)

        minimalPeakHeightForSearching = np.max(
            np.abs(self.minimalRelativeHeight * spanY),
            np.abs(noiseLevel * self.minimalSignalToNoiseRatio),
        )

        minimalFWHM = (
            self.minimalFWHMValue
            if self.isMinimalFWHMValueInXUnits and self.minimalFWHMValue > 0
            else np.max(1, self.minimalFWHMValue) * minIncrement
        )
        maximalFWHM = spanX

        parameters = lmfit.Parameters()
        fitFunction.AddBaselineParameters(parameters)

        for numberOfTerms in range(1, self.maximumNumberOfPeaks + 1):
            (peakIndices, peakProperties) = signal.find_peaks(
                yRest, height=minimalPeakHeightForSearching, width=0.0, rel_height=0.5
            )
            peakHeights = peakProperties["peak_heights"]
            peakWidths = signal.peak_widths(yRest, peakIndices, 0.5)
            # Sort the three arrays by peak height
            sortKey = np.argsort(peakHeights)
            peakHeights = peakHeights[sortKey]
            peakWidths = peakWidths[sortKey]
            peakIndices = peakIndices[sortKey]

            thisPeakIndex = -1
            for j in range(len(peakHeights), -1, -1):
                thisPeakIndex = j
                break

            if thisPeakIndex < 0 or peakHeights[thisPeakIndex] == 0:
                break  # no more peaks to fit

            thisPeakPosition = peakIndices[thisPeakIndex]

            yMax = y[thisPeakPosition]
            fwhm = 0.5 * np.abs(
                GetWidthValue(
                    x,
                    thisPeakPosition - 0.5 * peakWidths[thisPeakIndex],
                    thisPeakPosition,
                    thisPeakPosition + 0.5 * peakWidths[thisPeakIndex],
                )
            )
            fitFunction = fitFunction.WithNumberOfTerms(numberOfTerms)
            # at first, set the previous parameters to fixed
            for para in parameters:
                para.vary = False
            fitFunc.AddInitialParametersFromHeightPositionAndWidthAtRelativeHeight(
                parameters, numberOfTerms - 1, yMax, x[thisPeakPosition], fwhm, 0.5
            )
            fitFunc.SetParameterBoundariesForPositivePeaks(
                minimalX, maximalX, minimalFWHM, maximalFWHM
            )
            min1 = lmfit.Minimizer(
                fitFunction.func, parameters, fcn_args=(x,), fcn_kws={"data": y}
            )
            out1 = min1.leastsq(Dfun=fitFunction.dfunc, col_deriv=1)

            # at second, set all parameters to vary
            for para in parameters:
                para.vary = True

            # now fit again, with all parameters to be able to vary
            out2 = min1.leastsq(Dfun=fitFunction.dfunc, col_deriv=1)

            # calculate the yRest as the difference between y and the fit
            yRest = y - fitFunc.func(parameters, x)
        # the parameters now contains all peaks
        # lets see if some of them can be pruned, because they do not contribute much to the improvement of the fit
        return parameters