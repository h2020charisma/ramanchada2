#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, List, Tuple

import pydantic
import numpy as np
import numpy.typing as npt

from .pydantic_base_model import PydBaseModel
from ..plottable import Plottable


class PeakCandidateModel(PydBaseModel, Plottable):
    peak_idx: int
    peak_pos: int
    prominence: float

    half_max_val: float

    left_base_idx: int
    left_base_pos: float
    left_base_val: float

    right_base_idx: int
    right_base_pos: float
    right_base_val: float

    left_whm_idx: int
    left_whm_pos: float

    right_whm_idx: int
    right_whm_pos: float

    @property
    def fwhm(self) -> float:
        return abs(self.left_whm_pos - self.right_base_pos)

    @property
    def fwhm_idx(self) -> float:
        return abs(self.left_whm_idx - self.right_base_idx)

    @property
    def sigma(self):
        return self.fwhm / 2.355

    @property
    def sigma_idx(self):
        return self.fwhm_idx / 2.355

    @property
    def peak_base(self) -> float:
        return max(self.left_base_val, self.right_base_val)

    @property
    def peak_val(self) -> float:
        return self.peak_base + self.prominence

    @property
    def peak_pos_cor(self) -> float:
        return (2*self.peak_pos + self.left_whm_pos + self.right_whm_pos)/4

    def boundaries(self, n_sigma):
        return (
            self.peak_pos_cor - n_sigma*self.sigma,
            self.peak_pos_cor + n_sigma*self.sigma,
        )

    def boundaries_idx(self, n_sigma, arr_len):
        return (
            round(max(self.peak_idx - n_sigma*self.sigma_idx, 0)),
            round(min(self.peak_idx + n_sigma*self.sigma_idx, arr_len)),
        )

    def pos_ampl_sigma_base_peakidx(self):
        amp = self.prominence
        ped = self.peak_base
        pos = self.peak_pos_cor
        sig = self.sigma
        return pos, amp, sig, ped, self.peak_idx

    def plot_params_baseline(self):
        return ((self.left_base_pos, self.right_base_pos), (self.left_base_val, self.right_base_val))

    def plot_params_errorbar(self):
        x = self.peak_pos_cor
        xleft = self.peak_pos_cor - self.left_whm_pos
        xright = self.right_whm_pos - self.peak_pos_cor
        y = self.half_max_val
        y_err = (self.prominence/2)
        return (x,), (y,), (y_err,), ((xleft,), (xright,))

    def _plot(self, ax, *args, label=" ", **kwargs):
        ax.errorbar(*self.plot_params_errorbar(), label=label)
        ax.plot(*self.plot_params_baseline())

    def baseline_linear(self):
        slope = (self.right_base_val - self.left_base_val)/(self.right_base_pos-self.left_base_pos)
        intercept = -slope + self.left_base_val
        return intercept, slope

    def serialize(self):
        self.json()


class PeakCandidatesGroupModel(PydBaseModel, Plottable):
    __root__: List[PeakCandidateModel]

    @pydantic.validator('__root__', pre=False)
    def post_validate(cls, val):
        return sorted(val, key=lambda x: x.peak_idx)

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_find_peaks(arg: Tuple[npt.NDArray, Dict[str, npt.NDArray]],
                        x_arr: npt.NDArray,
                        y_arr: npt.NDArray,
                        ):
        peaks = arg[0]
        properties = arg[1]

        properties.update({'peak': peaks})
        properties_pivot = [dict(zip(properties, i)) for i in zip(*properties.values())]

        def straight_line(x):
            x1 = int(x)
            x2 = x1 + 1
            y1 = x_arr[x1]
            y2 = x_arr[x2]
            return (y2-y1)/(x2-x1)*(x-x1)+y1

        peak_list = [PeakCandidateModel(
            peak_idx=prop['peak'],
            peak_pos=x_arr[prop['peak']],
            prominence=prop['prominences'],
            half_max_val=prop['width_heights'],
            left_base_idx=prop['left_bases'],
            left_base_pos=x_arr[prop['left_bases']],
            left_base_val=y_arr[prop['left_bases']],
            right_base_idx=prop['right_bases'],
            right_base_pos=x_arr[prop['right_bases']],
            right_base_val=y_arr[prop['right_bases']],

            left_whm_idx=round(prop['left_ips']),
            left_whm_pos=straight_line(prop['left_ips']),

            right_whm_idx=round(prop['right_ips']),
            right_whm_pos=straight_line(prop['right_ips'])
        )
            for prop in properties_pivot]
        return PeakCandidatesGroupModel.validate(peak_list)

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_find_peaks_bayesian_gaussian_mixture(means, sigmas, weights,
                                                  x_arr: npt.NDArray,
                                                  y_arr: npt.NDArray,
                                                  ):
        #TODO
        peak_list = [PeakCandidateModel(
            peak_idx=x_arr[np.argmin((x_arr-mean)**2)],
            peak_pos=mean,
            prominence=weight,
            half_max_val=prop['width_heights'],
            left_base_idx=0,
            left_base_pos=x_arr[prop['left_bases']],
            left_base_val=y_arr[prop['left_bases']],
            right_base_idx=0,
            right_base_pos=x_arr[prop['right_bases']],
            right_base_val=y_arr[prop['right_bases']],

            left_whm_idx=round(prop['left_ips']),
            left_whm_pos=straight_line(prop['left_ips']),

            right_whm_idx=round(prop['right_ips']),
            right_whm_pos=straight_line(prop['right_ips'])
        )
            for prop in properties_pivot]

        return PeakCandidatesGroupModel.validate(peak_list)

    @property
    def peak_vals(self) -> List[float]:
        return [i.peak_val for i in self.__root__]

    @property
    def left_base_idx(self):
        return self.__root__[0].left_base_idx

    @property
    def right_base_idx(self):
        return self.__root__[-1].right_base_idx

    def left_base_idx_range(self, n_sigma, arr_len):
        w = self.__root__[0].sigma
        left = round(self.__root__[0].peak_idx - n_sigma*w)
        if left < 0:
            left = 0
        if left >= arr_len:
            left = arr_len - 1
        return left, self.__root__[0].peak_idx

    def right_base_idx_range(self, n_sigma, arr_len):
        w = self.__root__[-1].sigma
        right = round(self.__root__[-1].peak_idx + n_sigma*w)
        if right >= arr_len:
            right = arr_len - 1
        return self.__root__[-1].peak_idx, right

    @property
    def peaks(self):
        return [i.peak_pos for i in self.__root__]

    @property
    def peak_pos_cors(self):
        return [i.peak_pos_cor for i in self.__root__]

    def sigmas(self, x_arr=None):
        return [i.sigma(x_arr=x_arr) for i in self.__root__]

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

    def pos_ampl_sigma_base_peakidx(self):
        return [i.pos_ampl_sigma_base_peakidx() for i in self.__root__]

    def serialize(self):
        return self.json()

    def __len__(self):
        return len(self.__root__)

    def __getitem__(self, key: int) -> PeakCandidateModel:
        return self.__root__[key]

    def __iter__(self):
        return iter(self.__root__)

    def boundaries_idx(self, n_sigma, arr_len):
        left, _ = self.__root__[0].boundaries_idx(n_sigma=n_sigma, arr_len=arr_len)
        _, right = self.__root__[-1].boundaries_idx(n_sigma=n_sigma, arr_len=arr_len)
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
