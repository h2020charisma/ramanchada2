#!/usr/bin/env python3

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from uncertainties import unumpy

from ramanchada2.spectral_components.spectral_component import SpectralComponent


class SpectralPeak(SpectralComponent, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _plot(self, ax, draw='line', **kwargs):
        if draw == 'line':
            x = np.arange(*self.limit_3sigma)
            y = self(x)
            ax.errorbar(unumpy.nominal_values(x), unumpy.nominal_values(y),
                        yerr=unumpy.std_devs(y), **kwargs)
        elif draw == 'delta':
            args = dict(basefmt='', markerfmt='rD')
            args.update(kwargs)
            ax.stem(*self.delta, **args)
        elif draw == 'cross':
            x0, a, fwhm = self.pos_amp_fwhm
            ax.errorbar(x0, a/2, xerr=fwhm/2, yerr=a/2, **kwargs)
        else:
            raise TypeError("draw can be 'line', 'cross' or 'delta'")

    @abstractmethod
    def delta(self): pass

    @abstractmethod
    def limit_3sigma(self): pass

    @abstractmethod
    def pos_amp_fwhm(self): pass
