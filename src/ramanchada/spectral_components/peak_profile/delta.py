#!/usr/bin/env python3

from __future__ import annotations
import numpy as np

from uncertainties import unumpy

from ..spectral_peak import SpectralPeak


class DeltasPeak(SpectralPeak):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if {'a', 'x0'} - kwargs.keys():
            raise ValueError("'a', 'x0' arguments required")
        self.a = kwargs['a']
        self.x0 = kwargs['x0']

    def __call__(self, x: unumpy.uarray):
        ret = np.zeros_like(x)
        ret[x == self.x0] = self.a
        return ret

    @property
    def delta(self):
        return self.x0, self.a

    @property
    def pos_amp_fwhm(self):
        return self.x0, self.a, 0

    @property
    def limit_3sigma(self):
        return self.x0-1, self.x0+1
