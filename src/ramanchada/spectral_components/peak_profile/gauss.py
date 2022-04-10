#!/usr/bin/env python3

from __future__ import annotations
import numpy as np

from uncertainties import unumpy

from ..spectral_peak import SpectralPeak


class GaussPeak(SpectralPeak):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if {'a', 'x0', 'w'} - kwargs.keys():
            raise ValueError("'a', 'x0', 'w' arguments required")
        self.a = kwargs['a']
        self.x0 = kwargs['x0']
        self.w = kwargs['w']

    def __call__(self, x: unumpy.uarray):
        ret = 1/np.sqrt(2*np.pi)/self.w * self.a * unumpy.exp(-(x-self.x0)**2/2/self.w**2)
        return ret

    @property
    def delta(self):
        return self.x0, self.a

    @property
    def pos_amp_fwhm(self):
        return self.x0, self.a/np.sqrt(2*np.pi)/self.w, self.w*2*np.sqrt(2*np.log(2))

    @property
    def limit_3sigma(self):
        return self.x0-3*self.w, self.x0+3*self.w
