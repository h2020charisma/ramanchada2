import re
from collections import UserList, defaultdict
from typing import List

import numpy as np
import pandas as pd
from lmfit.model import Model, ModelResult, Parameters

from ..plottable import Plottable
from .peak_candidates import ListPeakCandidateMultiModel
from ramanchada2.spectrum.spectrum import Spectrum


class FitPeaksResult(UserList, Plottable):
    def valuesdict(self):
        ret = dict()
        for i in self:
            ret.update(i.params.valuesdict())
        return ret

    @property
    def locations(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('center')])

    @property
    def centers(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('center')])

    @property
    def fwhm(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('fwhm')])

    def boundaries(self, n_sigma=5):
        bounds = list()
        for group in self:
            pos = np.array([v for k, v in group.values.items() if k.endswith('center')])
            sig = np.array([v for k, v in group.values.items() if k.endswith('fwhm')])
            sig /= 2.35
            sig *= n_sigma
            bounds.append((np.min(pos - sig), np.max(pos+sig)))
        return bounds

    def center_amplitude(self, threshold):
        return np.array([
            (v.value, peak.params[k[:-6] + 'amplitude'].value)
            for peak in self
            for k, v in peak.params.items()
            if k.endswith('center')
            if hasattr(v, 'stderr') and v.stderr is not None and v.stderr < threshold
        ]).T

    @property
    def centers_err(self):
        return np.array([
            (v.value, v.stderr)
            for peak in self
            for k, v in peak.params.items()
            if k.endswith('center')
            if hasattr(v, 'stderr') and v.stderr is not None
            ])

    @property
    def fwhms(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('fwhm')])

    @property
    def amplitudes(self):
        return np.array([v for peak in self for k, v in peak.values.items() if k.endswith('amplitude')])

    def dumps(self):
        return [peak.dumps() for peak in self]

    @classmethod
    def loads(cls, json_str: List[str]):
        self = cls()
        for p in json_str:
            params = Parameters()
            modres = ModelResult(Model(lambda x: x, None), params)
            self.append(modres.loads(p))
        return self

    def _plot(self, ax, peak_candidate_groups=None, individual_peaks=False, xarr=None,
              label=None,  # ignore label from kwargs
              **kwargs):
        def group_plot(x, fitres, label=''):
            if individual_peaks:
                color = None
                for component in fitres.eval_components(x=x).values():
                    line, = ax.plot(x, component, color=color,
                                    label=(label if color is None else None),
                                    **kwargs)
                    color = line.get_c()
            else:
                ax.plot(x, fitres.eval(x=x), **kwargs)

        if peak_candidate_groups is None:
            for group_i, (bound, fitres) in enumerate(zip(self.boundaries(), self)):
                x = np.linspace(*bound, 200)
                group_plot(x, fitres, label=f'group {group_i}')
        elif isinstance(peak_candidate_groups, ListPeakCandidateMultiModel):
            for group_i, (cand, fitres) in enumerate(zip(peak_candidate_groups, self)):
                x = np.linspace(*cand.boundaries, 2000)
                group_plot(x, fitres, label=f'group {group_i}')
        else:
            for group_i, fitres in enumerate(self):
                left, right = peak_candidate_groups[group_i].boundaries(n_sigma=3)
                x = np.linspace(left, right, 100)
                group_plot(x, fitres, label=f'group {group_i}')

    def to_dataframe(self):
        return pd.DataFrame(
            [
                dict(name=f'g{group:02d}_{key}', value=val.value, stderr=val.stderr)
                for group, res in enumerate(self)
                for key, val in res.params.items()
            ]
        ).sort_values('name')

    def to_dataframe_peaks(self):
        regex = re.compile(r'p([0-9]+)_(.*)')
        ret = defaultdict(dict)
        for group_i, group in enumerate(self):
            for par in group.params:
                m = regex.match(par)
                if m is None:
                    continue
                peak_i, par_name = m.groups()
                ret[f'g{group_i:02d}_p{peak_i}'][par_name] = group.params[par].value
                ret[f'g{group_i:02d}_p{peak_i}'][f'{par_name}_stderr'] = group.params[par].stderr
        return pd.DataFrame.from_dict(ret, orient='index')

    def to_csv(self, path_or_buf=None, sep=',', **kwargs):
        return self.to_dataframe_peaks().to_csv(path_or_buf=path_or_buf, sep=sep, **kwargs)

    def gen_fake_spectrum(self, xarr):
        summ = np.zeros_like(xarr)
        last_i = 0
        last_y = 0
        for m in self:
            mx = m.userkws['x']
            xi = np.searchsorted(xarr, mx)
            evalm = m.eval()
            summ[xi] = evalm
            summ[np.arange(last_i, xi[0])] = np.interp(np.arange(last_i, xi[0]), [last_i, xi[0]], [last_y, evalm[0]])
            last_i = xi[-1]
            last_y = evalm[-1]
        fake_spe = Spectrum(x=xarr, y=summ)
        return fake_spe
