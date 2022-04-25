#!/usr/bin/env python3

from typing import Literal, List, Tuple

import pydantic

from .crystal.discrete_lines_out import lines_from_crystal_out
from .vasp.vasp_simulation_dat import lines_from_vasp_dat


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def read_simulated_lines(data_in,
                         sim_type: Literal['vasp', 'crystal'],
                         ) -> Tuple[List[str], List[float], List[float]]:
    if sim_type == 'crystal':
        tbl = lines_from_crystal_out(data_in)
        names = [f'{o}_{ml}_{mu}' for o, ml, mu in tbl[['Origin', 'ModeL', 'ModeU']].values]
        intensities = tbl['I_tot'].to_list()
        positions = tbl['Energy'].to_list()

    elif sim_type == 'vasp':
        tbl = lines_from_vasp_dat(data_in)
        names = [f'_{mode}'.replace('.', '_') for mode in tbl['mode']]
        intensities = tbl['activity'].to_list()
        positions = tbl['freq(cm-1)'].to_list()
    else:
        raise Exception('This should never happen')
    return names, intensities, positions
