#!/usr/bin/env python3

from __future__ import annotations
from typing import Set, Union, Dict, List, Tuple
from copy import deepcopy
import logging

import numpy as np
import numpy.typing as npt
import pydantic
from scipy.stats import rv_histogram

from ramanchada2.misc.plottable import Plottable
from ramanchada2.misc.types import SpeMetadataModel
from ramanchada2.misc.types.spectrum import (SpeProcessingListModel,
                                             SpeProcessingModel)
from ramanchada2.io.HSDS import write_cha
from ramanchada2.io.output.write_csv import write_csv as io_write_csv

logger = logging.getLogger(__name__)


class Spectrum(Plottable):
    _available_processings: Set[str] = set()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self,
                 x: Union[npt.NDArray, int, None] = None,
                 y: Union[npt.NDArray, None] = None,
                 cachefile: str = '',
                 metadata: Union[SpeMetadataModel, None] = None,
                 applied_processings: Union[SpeProcessingListModel, None] = None):
        super(Plottable, self).__init__()
        if x is not None:
            if isinstance(x, int):
                self.x = np.arange(x)
            else:
                self.x = x
        if y is not None:
            self.y = y

        sort_idx = np.argsort(self.x)
        if (np.diff(sort_idx) != 1).any():
            self.x = self.x[sort_idx]
            if self.y is not None:
                self.y = self.y[sort_idx]

        self._cachefile = cachefile
        self._metadata = deepcopy(metadata or SpeMetadataModel(__root__={}))
        self._applied_processings = deepcopy(applied_processings or SpeProcessingListModel(__root__=[]))

    def __copy__(self):
        return Spectrum(
            x=self.x,
            y=self.y,
            cachefile=self._cachefile,
            metadata=self._metadata,
            applied_processings=self._applied_processings,
        )

    def __repr__(self):
        return self._applied_processings.repr()

    def __str__(self):
        return str(self._applied_processings.to_list())

    def write_csv(self, filename, delimiter=',', newline='\n'):
        csv = io_write_csv(self.x, self.y, delimiter=delimiter)
        with open(filename, 'w', newline=newline) as f:
            for c in csv:
                f.write(c)

    def write_cha(self, chafile, dataset):
        write_cha(chafile, dataset,
                  self.x, self.y, self.meta.serialize())

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
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def apply_creator(cls, step: SpeProcessingModel, cachefile_=None):
        proc = getattr(cls, step.proc)
        spe = proc(*step.args, **step.kwargs, cachefile_=cachefile_)
        return spe

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
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
        self.x = self.x[idx]
        self.y = self.y[idx]

    @property
    def x(self): return np.array(self._xdata)

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
        return np.array(self._ydata)

    @y.setter
    def y(self, val: npt.NDArray[np.float64]):
        self._ydata = val
        self._ydata.flags.writeable = False

    @property
    def y_noise(self):
        return np.std(np.sort(np.diff(self.y))[len(self.y)//4:len(self.y)*3//4])

    @property
    def x_err(self):
        return np.zeros_like(self._xdata)

    @property
    def y_err(self):
        return np.zeros_like(self._ydata)

    @property
    def meta(self) -> SpeMetadataModel:
        return self._metadata

    @property
    def result(self):
        return self.meta['ramanchada2_filter_result']

    @result.setter
    def result(self, res: Union[Dict, List]):
        return self.meta._update(dict(ramanchada2_filter_result=res))

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def spe_distribution(self, trim_range: Union[Tuple[float, float], None] = None):
        x_all = self.x_bin_boundaries
        if trim_range is not None:
            l_idx = int(np.argmin(np.abs(x_all - trim_range[0])))
            r_idx = int(np.argmin(np.abs(x_all - trim_range[1])))
            spe_dist = rv_histogram((self.y[l_idx:r_idx], x_all[l_idx:r_idx+1]))
        else:
            spe_dist = rv_histogram((self.y, x_all))
        return spe_dist

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def gen_samples(self, size: pydantic.PositiveInt, trim_range=None):
        spe_dist = self.spe_distribution(trim_range=trim_range)
        samps = spe_dist.rvs(size=size)
        return samps
