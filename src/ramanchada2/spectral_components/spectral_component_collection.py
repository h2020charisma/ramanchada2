#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from ramanchada2.misc.base_class import BaseClass
from ramanchada2.misc.plottable import Plottable
from .spectral_peak import SpectralPeak


class SpectralComponentCollection(Plottable, BaseClass):
    def __init__(self, peaks, **kwargs):
        super(Plottable, self).__init__()
        super(BaseClass, self).__init__()
        self._peaks = set(peaks)
        self.kwargs = kwargs
        self.reset_origin()

    def reset_origin(self):
        self._origin = [(type(self).__name__,
                         [tuple(sorted(self._peaks, key=lambda x: repr(x)))],
                         self.kwargs)]

    def __call__(self, x):
        ret = np.array([p(x) for p in self._peaks]).sum(axis=0)
        return ret

    def get_deltas(self):
        pos, ampl = zip(*[p.delta for p in self._peaks])
        return pos, ampl

    def get_curve(self):
        ...

    @property
    def limit_3sigma(self):
        lims = [p.limit_3sigma for p in self._peaks]
        return np.min(lims), np.max(lims)

    def __iadd__(self, peak: SpectralPeak):
        self._peaks.add(peak)
        self.reset_origin()

    def _plot(self, ax, draw='combined line', **kwargs):
        if draw == 'deltas':
            stem_kwargs = dict(basefmt='', markerfmt='rD')
            stem_kwargs.update(kwargs)
            ax.stem(*self.get_deltas(), **stem_kwargs)
        elif draw == 'crosses':
            x0, a, fwhm = zip(*[i.pos_amp_fwhm for i in self._peaks])
            x0 = np.array(x0)
            a = np.array(a)
            fwhm = np.array(fwhm)
            err_kwargs = dict(linewidth=0, elinewidth=1)
            err_kwargs.update(kwargs)
            ax.errorbar(x0, a/2, xerr=fwhm/2, yerr=a/2, **err_kwargs)
        elif draw == 'combined line':
            x = np.arange(*self.limit_3sigma)
            ax.plot(x, self(x), **kwargs)
        elif draw == 'individual lines':
            for p in self._peaks:
                p.plot(ax, draw='line', **kwargs)
        else:
            raise TypeError("draw can be 'combined line', 'individual lines', 'crosses' or 'deltas'")
