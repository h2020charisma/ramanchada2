from __future__ import annotations

from typing import List, Tuple

import numpy as np
from pydantic import PositiveFloat

from ..plottable import Plottable
from .pydantic_base_model import PydBaseModel, PydRootModel


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
        return self.model_dump_json()


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
    def sigmas(self):
        return np.array([p.sigma for p in self.peaks])

    @property
    def fwhms(self):
        return np.array([p.fwhm for p in self.peaks])

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

    def get_pos_ampl_dict(self):
        return dict(zip(self.positions, self.amplitudes))

    def get_ampl_pos_fwhm(self):
        return np.array([
            self.amplitudes,
            self.positions,
            self.fwhms]).T

    @property
    def peak_bases(self):
        return self.positions * self.base_slope + self.base_intercept

    def _plot(self, ax, *args, label=" ", **kwargs):
        ax.errorbar(*self.plot_params_errorbar(), label=label)
        ax.plot(*self.plot_params_baseline())

    def serialize(self):
        return self.model_dump_json()


class ListPeakCandidateMultiModel(PydRootModel, Plottable):
    root: List[PeakCandidateMultiModel]

    def get_ampl_pos_fwhm(self):
        return np.concatenate([cands.get_ampl_pos_fwhm() for cands in self.root])

    def get_pos_ampl_dict(self):
        return {k: v for cands in self.root for k, v in cands.get_pos_ampl_dict().items()}

    def _plot(self, ax, *args, label=" ", **kwargs):
        for i, gr in enumerate(self.root):
            gr.plot(ax=ax, *args, label=f'{label}_{i}', **kwargs)

    def __getitem__(self, key) -> PeakCandidateMultiModel:
        return self.root[key]

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)

    def serialize(self):
        return self.model_dump_json()
