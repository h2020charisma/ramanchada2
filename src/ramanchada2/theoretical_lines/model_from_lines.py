#!/usr/bin/env python3

from typing import List, Tuple, Literal

import numpy as np
from lmfit import Parameters, Model
from lmfit.models import VoigtModel, GaussianModel

from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def model_from_lines(names: List[str],
                     intensities: List[float],
                     positions: List[float],
                     model: Literal['gaussian', 'voigt'] = 'gaussian'
                     ) -> Tuple[Model, Parameters]:

    if model == 'gaussian':
        lm_model = GaussianModel
    elif model == 'voigt':
        lm_model = VoigtModel
    else:
        raise ValueError(f'model {model} not known')
    mod = np.sum([lm_model(prefix=f"{name}_", name=name) for name in names])

    params = Parameters()
    params.add('amplitude', 1, min=0)
    params.add('pedestal', 0, min=0)
    params.add('sigma', 10, min=0)
    params.add('x0', 0)
    params.add('xscale', 1)
    for name, intens, pos in zip(names, intensities, positions):
        prefix = f"{name}_"
        params.add(prefix+'amplitude', expr=f"({intens}*amplitude)+pedestal")
        params.add(prefix+'center', expr=f"({pos}*xscale)+x0")
        params.add(prefix+'sigma', expr='sigma')

    return mod, params
