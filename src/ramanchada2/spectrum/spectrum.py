#!/usr/bin/env python3

from __future__ import annotations
from typing import Set
from types import MappingProxyType


import numpy as np
import numpy.typing as npt
from uncertainties import unumpy

from ramanchada2.misc.plottable import Plottable
from ramanchada2.misc.base_class import BaseClass
from ramanchada2.misc.types import SpectrumMetaData


class Spectrum(Plottable, BaseClass):
    _available_processings: Set[str] = set()

    def __init__(self, **kwargs):
        super(Plottable, self).__init__()
        super(BaseClass, self).__init__()
        self.cachedir = kwargs['cachedir'] if 'cachedir' in kwargs else None
        self.h5file = kwargs['h5file'] if 'h5file' in kwargs else None
    __copy__elements = ['x', 'y', 'meta', 'cachedir', 'h5file', 'origin']

    def __copy__(self):
        ret = Spectrum()
        for elem in self.__copy__elements:
            setattr(ret, elem, getattr(self, elem))
        return ret

    def commit(self):
        if self.cachedir:
            pass

    def process(self, algorithm: str, **kwargs):
        if algorithm not in self._available_processings:
            raise ValueError('Unknown algorithm {algorithm}')
        return getattr(self, algorithm)(**kwargs)

    def _plot(self, ax, *args, **kwargs):
        ax.errorbar(
            self.x,
            self.y,
            xerr=self.x_err,
            yerr=self.y_err,
            **kwargs
        )

    @property
    def x(self): return unumpy.nominal_values(self._xdata)

    @x.setter
    def x(self, val: npt.NDArray[np.float64]):
        self._xdata = val
        self._xdata.flags.writeable = False

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return unumpy.nominal_values(self._ydata)

    @y.setter
    def y(self, val: npt.NDArray[np.float64]):
        self._ydata = val
        self._ydata.flags.writeable = False

    @property
    def x_err(self):
        return np.zeros_like(self._xdata)

    @property
    def y_err(self):
        return np.zeros_like(self._ydata)

    @property
    def meta(self):
        return self._metadata

    @meta.setter
    def meta(self, val: SpectrumMetaData):
        self._metadata = MappingProxyType(val)
