#!/usr/bin/env python3

from typing import List, Tuple, Literal, Dict

import numpy as np
from lmfit import Parameters, Model
from lmfit.models import VoigtModel, GaussianModel

from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
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
    params.add('x1', 1)

    for spe_type, spe_int in intensities.items():
        spe_prefix = f'{spe_type}_'
        params.add(spe_prefix+'amplitude', 1, min=0)
        for name, pos, line_int in zip(names, positions, spe_int):
            line_prefix = f'{spe_prefix}{name}_'
            params.add(line_prefix+'amplitude', expr=f'({line_int}*{spe_prefix}amplitude)+pedestal')
            params.add(line_prefix+'center', expr=f'({pos}*x1)+x0')
            params.add(line_prefix+'sigma', expr='sigma')

    return mod, params
