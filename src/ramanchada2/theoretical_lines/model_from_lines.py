from typing import Dict, List, Literal, Tuple

import numpy as np
from lmfit import Model, Parameters
from lmfit.models import GaussianModel, VoigtModel
from pydantic import BaseModel, Field, validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def model_from_lines(names: List[str],
                     positions: List[float],
                     intensities: Dict[str, List[float]],
                     model: Literal['gaussian', 'voigt'] = 'gaussian'
                     ) -> Tuple[Model, Parameters]:

    if model == 'gaussian':
        lm_model = GaussianModel
    elif model == 'voigt':
        lm_model = VoigtModel
    else:
        raise ValueError(f'model {model} not known')
    mod = np.sum([
        lm_model(prefix=f'{spe_type}_{name}_', name=f'{spe_type}_{name}')
        for spe_type in intensities
        for name in names
        ])

    params = Parameters()
    params.add('pedestal', 0, min=0)
    params.add('sigma', 2, min=0)
    params.add('x0', 0)
    params.add('x1', 1, min=0)

    for spe_type, spe_int in intensities.items():
        spe_prefix = f'{spe_type}_'
        params.add(spe_prefix+'amplitude', 1, min=0)
        for name, pos, line_int in zip(names, positions, spe_int):
            line_prefix = f'{spe_prefix}{name}_'
            params.add(line_prefix+'amplitude',
                       expr=f'({line_int}*{spe_prefix}amplitude)+pedestal')
            params.add(line_prefix+'center', expr=f'(({pos}+x0)*x1)')
            params.add(line_prefix+'sigma', expr='sigma')

    return mod, params


class PydPeakModel(BaseModel):
    model: Literal['gaussian', 'voigt'] = Field('voigt')
    position: float
    inensity: float = Field(1, gt=0)
    sigma: float = Field(1, gt=0)
    name: str = Field('')


@validate_call(config=dict(arbitrary_types_allowed=True))
def model_from_list(peaks_list: List[PydPeakModel]
                    ) -> Tuple[Model, Parameters]:
    params = Parameters()
    params.add('amplitude', 1, min=0)
    params.add('sigma', 1, min=0)
    params.add('x0', 0)
    params.add('x1', 1, min=0)
    params.add('x2', 0, min=-1e-3, max=1e-3)
    params.add('x3', 0, min=-1e-5, max=1e-5)

    peaks = list()
    for ii, peak in enumerate(peaks_list):
        if peak.model == 'gaussian':
            lm_model = GaussianModel
        elif peak.model == 'voigt':
            lm_model = VoigtModel
        else:
            raise ValueError(f'model {peak.model} not known')
        prefix = f'{peak.name}_' if peak.name else f'_{ii}_'
        name = f'{peak.name}' if peak.name else f'_{ii}'
        peaks.append(lm_model(prefix=prefix, name=name))

        params.add(prefix+'amplitude', expr=f'({peak.inensity}*amplitude)')
        params.add(prefix+'center',
                   expr=f'{peak.position}**3*x3 + {peak.position}**2*x2 + {peak.position}*x1 + x0')
        params.add(prefix+'sigma', expr=f'{peak.sigma}*sigma')
    mod = np.sum(peaks)

    return mod, params
