from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydantic import PositiveInt, validate_call
from scipy.signal import convolve, savgol_coeffs, savgol_filter
from scipy.stats import median_abs_deviation, rv_histogram

from ramanchada2.io.HSDS import write_cha, write_nexus
from ramanchada2.io.output.write_csv import write_csv as io_write_csv
from ramanchada2.misc.plottable import Plottable
from ramanchada2.misc.types import PositiveOddInt, SpeMetadataModel
from ramanchada2.misc.types.spectrum import (SpeProcessingListModel,
                                             SpeProcessingModel)

logger = logging.getLogger(__name__)


class Spectrum(Plottable):
    _available_processings: Set[str] = set()

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self,
                 x: Union[npt.NDArray, int, None] = None,
                 y: Union[npt.NDArray, None] = None,
                 cachefile: Optional[str] = None,
                 metadata: Union[SpeMetadataModel, None] = None,
                 applied_processings: Union[SpeProcessingListModel, None] = None):
        super(Plottable, self).__init__()
        self._xdata = None
        self._ydata = None
        if x is not None:
            if isinstance(x, int):
                self.x = np.arange(x) * 1.
            else:
                if x.dtype != float:
                    self.x = x.astype(float)
                else:
                    self.x = x
        if y is not None:
            if y.dtype != float:
                self.y = y.astype(float)
            else:
                self.y = y

        self._x_err: Union[npt.NDArray, None] = None
        self._y_err: Union[npt.NDArray, None] = None

        self._cachefile = cachefile
        self._metadata = deepcopy(metadata or SpeMetadataModel(root={}))
        self._applied_processings = deepcopy(applied_processings or SpeProcessingListModel(root=[]))
        if self._xdata is not None and self._ydata is not None:
            if len(self._xdata) != len(self._ydata):
                raise ValueError(
                    f'x and y shold have same dimentions len(x)={len(self._xdata)} len(y)={len(self._ydata)}')

    def __copy__(self):
        return Spectrum(
            x=self._xdata,
            y=self._ydata,
            cachefile=self._cachefile,
            metadata=self._metadata,
            applied_processings=self._applied_processings,
        )

    def __repr__(self):
        return self._applied_processings.repr()

    def applied_processings_dict(self):
        return self._applied_processings.to_list()

    def __str__(self):
        return str(self._applied_processings.to_list())

    def write_csv(self, filename, delimiter=',', newline='\n'):
        csv = io_write_csv(self.x, self.y, delimiter=delimiter)
        with open(filename, 'w', newline=newline) as f:
            for c in csv:
                f.write(c)

    def write_cha(self, chafile, dataset):
        write_cha(chafile, dataset, self.x, self.y, self.meta.serialize())

    def write_nexus(self, chafile, dataset):
        write_nexus(chafile, dataset, self.x, self.y, self.meta.serialize())

    def write_cache(self):
        if self._cachefile:
            self.write_cha(
                self._cachefile,
                '/cache/'+self._applied_processings.cache_path()+'/_data')

    def process(self, algorithm: str, **kwargs):
        if algorithm not in self._available_processings:
            raise ValueError('Unknown algorithm {algorithm}')
        return getattr(self, algorithm)(**kwargs)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def apply_creator(cls, step: SpeProcessingModel, cachefile_=None):
        proc = getattr(cls, step.proc)
        spe = proc(*step.args, **step.kwargs, cachefile_=cachefile_)
        return spe

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def apply_processing(self, step: SpeProcessingModel):
        proc = getattr(self, step.proc)
        spe = proc(*step.args, **step.kwargs)
        return spe

    def _plot(self, ax, *args, **kwargs):
        ax.errorbar(
            self.x,
            self.y,
            xerr=self.x_err,
            yerr=self.y_err,
            **kwargs
        )

    def _sort_x(self):
        idx = np.argsort(self.x)
        if (np.diff(idx) != 1).any():
            self.x = self.x[idx]
            self.y = self.y[idx]

    @property
    def x(self):
        if self._xdata is None:
            raise ValueError('x of the spectrum is not set. self._xdata is None')
        return np.array(self._xdata)

    @x.setter
    def x(self, val: npt.NDArray[np.float64]):
        self._xdata = val
        self._xdata.flags.writeable = False

    @property
    def x_bin_boundaries(self):
        return np.concatenate((
            [(3*self.x[0] - self.x[1])/2],
            (self.x[1:] + self.x[:-1])/2,
            [(3*self.x[-1] - self.x[-2])/2]
        ))

    @property
    def y(self) -> npt.NDArray[np.float64]:
        if self._ydata is None:
            raise ValueError('y of the spectrum is not set. self._ydata is None')
        return np.array(self._ydata)

    @y.setter
    def y(self, val: npt.NDArray[np.float64]):
        self._ydata = val
        self._ydata.flags.writeable = False

    @property
    def y_noise(self):
        return self.y_noise_savgol()

    def y_noise_MAD(self):
        return median_abs_deviation(np.diff(self.y))

    @validate_call(config=dict(validate_default=True))
    def y_noise_savgol_DL(self, order: PositiveOddInt = 1):
        npts = order + 2
        ydata = self.y - np.min(self.y)
        summ = np.sum((ydata - savgol_filter(ydata, npts, order))**2)
        coeff = savgol_coeffs(npts, order)
        coeff[(len(coeff) - 1) // 2] -= 1
        scale = np.sqrt(np.sum(coeff**2))
        return np.sqrt(summ/len(ydata))/scale

    @validate_call(config=dict(validate_default=True))
    def y_noise_savgol(self, order: PositiveOddInt = 1):
        npts = order + 2

        # subtract smoothed signal from original
        coeff = - savgol_coeffs(npts, order)
        coeff[(len(coeff)-1)//2] += 1

        # normalize coefficients so that `sum(coeff**2) == 1`
        coeff /= np.sqrt(np.sum(coeff**2))

        # remove the common floor
        ydata = self.y - np.min(self.y)
        return np.std(convolve(ydata, coeff, mode='same'))

    @property
    def x_err(self):
        if self._x_err is None:
            return np.zeros_like(self._xdata)
        else:
            return self._x_err

    @x_err.setter
    def x_err(self, val: Union[npt.NDArray, None]):
        if self._xdata is None:
            raise ValueError('x of the spectrum is not set. self._xdata is None')
        if val is not None:
            if val.shape != self._xdata.shape:
                raise ValueError(
                    'x_err should have same shape as xdata, expected {self._xdata.shape}, got {val.shape}')
        self._x_err = val

    @property
    def y_err(self):
        if self._y_err is None:
            return np.zeros_like(self._ydata)
        else:
            return self._y_err

    @y_err.setter
    def y_err(self, val: Union[npt.NDArray, None]):
        if self._ydata is None:
            raise ValueError('y of the spectrum is not set. self._ydata is None')
        if val is not None:
            if val.shape != self._ydata.shape:
                raise ValueError(
                    'y_err should have same shape as ydata, expected {self._ydata.shape}, got {val.shape}')
        self._y_err = val

    @property
    def meta(self) -> SpeMetadataModel:
        return self._metadata

    @meta.setter
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def meta(self, val: Union[Dict, SpeMetadataModel]):
        if isinstance(val, dict):
            self._metadata = SpeMetadataModel.model_validate(val)
        else:
            self._metadata = val

    @property
    def result(self):
        return self.meta['ramanchada2_filter_result']

    @result.setter
    def result(self, res: Union[Dict, List]):
        return self.meta._update(dict(ramanchada2_filter_result=res))

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def spe_distribution(self, trim_range: Union[Tuple[float, float], None] = None):
        x_all = self.x_bin_boundaries
        if trim_range is not None:
            l_idx = int(np.argmin(np.abs(x_all - trim_range[0])))
            r_idx = int(np.argmin(np.abs(x_all - trim_range[1])))
            spe_dist = rv_histogram((self.y[l_idx:r_idx], x_all[l_idx:r_idx+1]))
        else:
            spe_dist = rv_histogram((self.y, x_all))
        return spe_dist

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def gen_samples(self, size: PositiveInt, trim_range=None):
        spe_dist = self.spe_distribution(trim_range=trim_range)
        samps = spe_dist.rvs(size=size)
        return samps
