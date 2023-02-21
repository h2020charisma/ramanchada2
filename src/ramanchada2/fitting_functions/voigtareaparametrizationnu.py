#!/usr/bin/env python3

import numpy as np
from scipy.special import voigt_profile, wofz


OneBySqrtLog4 = 0.84932180028801904272150283410289      # 1/sqrt(log(4))
Log4 = 1.3862943611198906188344642429164                # Log(4)
Sqrt2 = 1.4142135623730950488016887242097               # Math.Sqrt(2)
Sqrt2Pi = 2.5066282746310005024157652848110             # Math.Sqrt(2*Math.Pi)


class VoigtAreaParametrizationNu:

    numberOfTerms = 1
    orderOfBaselinePolynomial = -1

    def __init__(self, numberOfTerms, orderOfBackgroundPolynomial):
        self.numberOfTerms = numberOfTerms
        self.orderOfBaselinePolynomial = orderOfBackgroundPolynomial

    def func(self, pars, x, data=None):
        sum = np.zeros(len(x))
        for i in range(self.orderOfBaselinePolynomial, -1, -1):
            sum *= x
            sum += pars[f'b{i}']

        for i in range(self.numberOfTerms):
            a, xc, w, nu = pars[f'area{i}'], pars[f'pos{i}'], pars[f'w{i}'], pars[f'nu{i}']
            sum += a * voigt_profile(x - xc, w * np.sqrt(nu) * OneBySqrtLog4, w * (1 - nu))

        if data is None:
            return sum
        return sum - data

    def dfunc(self, pars, x, data=None):
        result = []
        k = 0

        # first the derivatives of the peak terms
        for i in range(self.numberOfTerms):
            area, xc, w, nu = pars[f'area{i}'], pars[f'pos{i}'], pars[f'w{i}'], pars[f'nu{i}']
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
                    dfdnu = area * body * (arg * arg * (3 - arg * arg * Log4) + Log4 - 1) / (onePlusArg2 * onePlusArg2 * Log4)
                elif nu <= 1:  # general case including nu==1 (which means gamma==0, i.e. pure Gaussian)
                    arg_gammaj = np.empty(len(arg), dtype=np.complex128)
                    arg_gammaj.real = arg
                    arg_gammaj.imag = gamma                                     # arg_gammaj is now complex(arg, gamma)
                    z = arg_gammaj / (Sqrt2 * sigma)
                    wOfZ = wofz(z)
                    body = wOfZ / (Sqrt2Pi * sigma)
                    dbodydz = (Sqrt2 * 1.0j - z * wOfZ * Sqrt2Pi) / (np.pi * sigma)        # Derivative of wOfZBySqrt2PiSigma w.r.t. z
                    dfdarea = np.real(body)                                                         # Derivative w.r.t. amplitude
                    dfdpos = -area * np.real(dbodydz) / (Sqrt2 * sigma)                             # Derivative w.r.t. position
                    dfdw = -area / w * np.real(dbodydz * arg / (Sqrt2 * sigma) + body)              # Derivative w.r.t. w
                    argwonepnu = np.empty(len(arg), dtype=np.complex128)
                    argwonepnu.real = arg
                    argwonepnu.imag = w * (1 + nu)                         # complex(arg, w * (1 + nu))
                    dfdnu = -area / (2 * nu) * np.real((dbodydz * argwonepnu / (Sqrt2 * sigma) + body))     # Derivative w.r.t. nu

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
