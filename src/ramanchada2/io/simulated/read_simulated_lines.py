from io import TextIOBase
from typing import Dict, List, Literal, Set, Tuple

from pydantic import validate_call

from .crystal.discrete_lines_dat import lines_from_crystal_dat
from .crystal.discrete_lines_out import lines_from_crystal_out
from .lines_from_raw_dat import lines_from_raw_dat
from .vasp.vasp_simulation_dat import lines_from_vasp_dat


@validate_call(config=dict(arbitrary_types_allowed=True))
def read_simulated_lines(data_in: TextIOBase,
                         sim_type: Literal['vasp', 'crystal_out', 'crystal_dat', 'raw_dat'],
                         use: Set[Literal[
                             'I_tot', 'I_perp', 'I_par',
                             'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz'
                             ]] = {'I_tot'}
                         ) -> Tuple[List[str], List[float], Dict[str, List[float]]]:
    positions: List[float]
    intensities: Dict[str, List[float]] = dict()
    if sim_type.startswith('crystal'):
        if sim_type.endswith('out'):
            tbl = lines_from_crystal_out(data_in)
        elif sim_type.endswith('dat'):
            tbl = lines_from_crystal_dat(data_in)
        else:
            raise Exception('This should never happen')
        names = [f'_{i}' for i in range(len(tbl['I_tot']))]
        for key in use:
            intensities.update({key: tbl[key].to_list()})
        positions = tbl['Frequencies'].to_list()

    elif sim_type == 'vasp':
        tbl = lines_from_vasp_dat(data_in)
        names = [f'_{mode}'.replace('.', '_') for mode in tbl['mode']]
        for key in use:
            if key in {'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz'}:
                raise ValueError('vasp simulation does not support monocrystal intensities')
            if key == 'I_tot':
                intensities.update({key: tbl['activity'].to_list()})
            else:
                intensities.update({key: tbl[key].to_list()})
        positions = tbl['freq(cm-1)'].to_list()

    elif sim_type == 'raw_dat':
        tbl = lines_from_raw_dat(data_in)
        names = [''] * len(tbl['I_tot'])
        for key in use:
            intensities.update({key: tbl[key].to_list()})
        positions = tbl['Frequencies'].to_list()
    else:
        raise Exception('This should never happen')
    return names, positions, intensities
