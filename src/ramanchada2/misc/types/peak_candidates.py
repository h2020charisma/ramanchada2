#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Tuple
from pydantic import PositiveFloat

import numpy as np

from .pydantic_base_model import PydBaseModel
from ..plottable import Plottable


class PeakModel(PydBaseModel):
    amplitude: PositiveFloat
    position: float
    sigma: PositiveFloat
    skew: float = 0

    @property
    def fwhm(self) -> float:
        return self.sigma * 2.355

    @property
    def lwhm(self) -> float:
        """
        Left width at half maximum.
        """
        return self.fwhm*(1-self.skew)/2

    @property
    def rwhm(self) -> float:
        """
        Right width at half maximum.
        """
        return self.fwhm*(1+self.skew)/2

    def serialize(self):
        self.json()


class PeakCandidateMultiModel(PydBaseModel, Plottable):
    peaks: List[PeakModel]
    base_slope: float = 0
    base_intercept: float = 0
    boundaries: Tuple[float, float]

    def plot_params_baseline(self):
        x = np.array(self.boundaries)
        return (x, self.base_slope*x + self.base_intercept)

    def plot_params_errorbar(self):
        x = self.positions
        xleft = self.lwhms
        xright = self.rwhms
        y_err = (self.amplitudes/2)
        y = y_err + self.peak_bases
        return x, y, y_err, (xleft, xright)

    @property
    def positions(self):
        return np.array([p.position for p in self.peaks])

    @property
    def lwhms(self):
        return np.array([p.lwhm for p in self.peaks])

    @property
    def rwhms(self):
        return np.array([p.rwhm for p in self.peaks])

    @property
    def skews(self):
        return np.array([p.skew for p in self.peaks])

    @property
    def amplitudes(self):
        return np.array([p.amplitude for p in self.peaks])

    @property
    def bases(self):
        return self.positions * self.base_slope + self.base_intercept

    @property
    def peak_bases(self):
        return self.positions * self.base_slope + self.base_intercept

    def _plot(self, ax, *args, label=" ", **kwargs):
        ax.errorbar(*self.plot_params_errorbar(), label=label)
        ax.plot(*self.plot_params_baseline())

    def serialize(self):
        self.json()


class ListPeakCandidateMultiModel(PydBaseModel, Plottable):
    __root__: List[PeakCandidateMultiModel]

    def _plot(self, ax, *args, label=" ", **kwargs):
        for i, gr in enumerate(self.__root__):
            gr.plot(ax=ax, *args, label=f'{label}_{i}', **kwargs)

    def __getitem__(self, key) -> PeakCandidateMultiModel:
        return self.__root__[key]

    def __iter__(self):
        return iter(self.__root__)

    def serialize(self):
        self.json()
