#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, List, Tuple
from pydantic import PositiveFloat

import pydantic
import numpy as np
import numpy.typing as npt

from .pydantic_base_model import PydBaseModel
from ..plottable import Plottable


class PeakCandidateModel(PydBaseModel, Plottable):
    base_slope: float = 0
    base_intercept: float = 0
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

    @property
    def left_sigma(self) -> float:
        return self.sigma*(1-self.skew)

    @property
    def right_sigma(self) -> float:
        return self.sigma*(1+self.skew)

    def boundaries(self, n_sigma):
        return (
            self.position - n_sigma*self.left_sigma,
            self.position + n_sigma*self.right_sigma,
        )

    def boundaries_idx(self, n_sigma, x_arr):
        lb, rb = self.boundaries(n_sigma)
        lb_idx = np.argmin(np.abs(x_arr - lb))
        rb_idx = np.argmin(np.abs(x_arr - rb))
        return (lb_idx, rb_idx)

    def pos_ampl_sigma_base(self):
        return (self.position,
                self.amplitude,
                self.sigma,
                self.peak_base)

    def plot_params_baseline(self):
        x = np.array([self.position - 4*self.sigma, self.position + 4*self.sigma])
        return (x, self.base_slope*x + self.base_intercept)

    def plot_params_errorbar(self):
        x = self.position
        xleft = self.lwhm
        xright = self.rwhm
        y_err = (self.amplitude/2)
        y = y_err + self.peak_base
        return (x,), (y,), (y_err,), ((xleft,), (xright,))

    @property
    def peak_base(self):
        return self.position*self.base_slope + self.base_intercept

    def _plot(self, ax, *args, label=" ", **kwargs):
        ax.errorbar(*self.plot_params_errorbar(), label=label)
        ax.plot(*self.plot_params_baseline())

    def serialize(self):
        self.json()


class PeakCandidatesGroupModel(PydBaseModel, Plottable):
    __root__: List[PeakCandidateModel]

    @pydantic.validator('__root__', pre=False)
    def post_validate(cls, val):
        return sorted(val, key=lambda x: x.position)

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_find_peaks(arg: Tuple[npt.NDArray, Dict[str, npt.NDArray]],
                        x_arr: npt.NDArray,
                        y_arr: npt.NDArray,
                        ):
        def interpolate(x):
            x1 = int(x)
            x2 = x1 + 1
            y1 = x_arr[x1]
            y2 = x_arr[x2]
            return (y2-y1)/(x2-x1)*(x-x1)+y1

        peaks = arg[0]
        properties = arg[1]
        properties.update({'peak': peaks})
        properties_pivot = [dict(zip(properties, i)) for i in zip(*properties.values())]

        peak_list = list()
        for prop in properties_pivot:
            pos_maximum = x_arr[prop['peak']]
            lwhm = pos_maximum - interpolate(prop['left_ips'])
            rwhm = interpolate(prop['right_ips']) - pos_maximum
            fwhm = lwhm + rwhm
            sigma = fwhm/2.355

            left_base_pos = x_arr[prop['left_bases']]
            left_base_val = y_arr[prop['left_bases']]
            right_base_pos = x_arr[prop['right_bases']]
            right_base_val = y_arr[prop['right_bases']]

            slope = (right_base_val - left_base_val)/(right_base_pos - left_base_pos)
            intercept = -slope*left_base_pos + left_base_val
            peak_list.append(dict(amplitude=prop['prominences'],
                                  position=pos_maximum + (rwhm - lwhm)/4,
                                  sigma=sigma,
                                  skew=(rwhm-lwhm)/(rwhm+lwhm),
                                  base_slope=slope,
                                  base_intercept=intercept,
                                  ))
        return PeakCandidatesGroupModel.validate(peak_list)

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_find_peaks_bayesian_gaussian_mixture(bgm,
                                                  x_arr: npt.NDArray,
                                                  y_arr: npt.NDArray,
                                                  ):
        bgm_peaks = [[mean[0], np.sqrt(cov[0][0]), weight]
                     for mean, cov, weight in
                     zip(bgm.means_, bgm.covariances_, bgm.weights_)]
        bgm_peaks = sorted(bgm_peaks, key=lambda x: x[2], reverse=True)
        integral = np.sum(y_arr)
        n_peaks = (np.round(bgm.weights_, 2) > 0).sum()
        bgm_peaks = bgm_peaks[:n_peaks]

        peak_list = list()
        for mean, sigma, weight in bgm_peaks:
            peak_list.append(dict(amplitude=weight*integral*2/sigma,
                                  position=mean,
                                  sigma=sigma,
                                  ))
        return PeakCandidatesGroupModel.validate(peak_list)

    def sigmas(self):
        return [i.sigma for i in self.__root__]

    @property
    def amplitudes(self) -> List[float]:
        return [i.amplitude for i in self.__root__]

    @property
    def positions(self):
        return [i.position for i in self.__root__]

    @property
    def prominences(self):
        return [i.prominence for i in self.__root__]

    @property
    def left_bases(self):
        return [i.left_base for i in self.__root__]

    @property
    def right_bases(self):
        return [i.right_base for i in self.__root__]

    @property
    def widths(self):
        return [i.width for i in self.__root__]

    @property
    def width_heights(self):
        return [i.width_height for i in self.__root__]

    @property
    def left_ips(self):
        return [i.left_ip for i in self.__root__]

    @property
    def right_ips(self):
        return [i.right_ip for i in self.__root__]

    @property
    def base_lines(self):
        return [i.base_line for i in self.__root__]

    def pos_ampl_sigma_base(self):
        return [i.pos_ampl_sigma_base() for i in self.__root__]

    def serialize(self):
        return self.json()

    def __len__(self):
        return len(self.__root__)

    def __getitem__(self, key: int) -> PeakCandidateModel:
        return self.__root__[key]

    def __iter__(self):
        return iter(self.__root__)

    def boundaries_idx(self, n_sigma, x_arr):
        left, _ = self.__root__[0].boundaries_idx(n_sigma=n_sigma, x_arr=x_arr)
        _, right = self.__root__[-1].boundaries_idx(n_sigma=n_sigma, x_arr=x_arr)
        return left, right

    def boundaries(self, n_sigma):
        left, _ = self.__root__[0].boundaries(n_sigma=n_sigma)
        _, right = self.__root__[-1].boundaries(n_sigma=n_sigma)
        return min(left, right), max(left, right)

    def group_neighbours(self, n_sigma) -> ListPeakCandidateGroupsModel:
        res: List = list()
        last_right = -np.infty
        for peak in self.__root__:
            cur_left, cur_right = peak.boundaries(n_sigma=n_sigma)
            if cur_left < last_right:
                res[-1].append(peak)
            else:
                res.append([peak])
            last_right = cur_right
        return ListPeakCandidateGroupsModel.validate(res)

    def plot_params_errorbar(self):
        dat = [i.plot_params_errorbar() for i in self.__root__]
        x = next(zip(*[i[0] for i in dat]))
        y = next(zip(*[i[1] for i in dat]))
        y_err = next(zip(*[i[2] for i in dat]))
        x_err = next(zip(*[i[3][0] for i in dat])), next(zip(*[i[3][1] for i in dat]))
        return x, y, y_err, x_err

    def plot_params_individual_baselines(self):
        dat = np.array([i.plot_params_baseline() for i in self.__root__])
        return np.transpose(dat, (1, 2, 0))

    def _plot(self, ax, label=' ', **kwargs):
        ax.errorbar(*self.plot_params_errorbar(), label=label, **kwargs)
        ax.plot(*self.plot_params_individual_baselines())


class ListPeakCandidateGroupsModel(PydBaseModel, Plottable):
    __root__: List[PeakCandidatesGroupModel]

    def _plot(self, ax, **kwargs):
        for p in self.__root__:
            p.plot(ax, **kwargs)

    def __getitem__(self, key) -> PeakCandidatesGroupModel:
        return self.__root__[key]

    def __iter__(self):
        return iter(self.__root__)

    def serialize(self):
        return self.json()
