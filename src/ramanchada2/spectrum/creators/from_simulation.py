#!/usr/bin/env python3
"""Create spectrum from simulation output files."""

from io import TextIOBase
from typing import Literal, Union

import numpy as np
from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor
from ramanchada2.io.simulated import read_simulated_lines


@add_spectrum_constructor()
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def from_simulation(in_file: Union[str, TextIOBase],
                    sim_type: Literal['vasp', 'crystal_out', 'crystal_dat', 'raw_dat'],
                    use: Literal['I_tot', 'I_perp', 'I_par',
                                 'I_xx', 'I_xy', 'I_xz',
                                 'I_yy', 'I_yz', 'I_zz'] = 'I_tot'
                    ):
    """
    Generate spectrum from simulation file.

    The returned spectrum has only few x/y pairs -- one for each simulated line. Values along
    the x-axis will not be uniform. To make it uniform, one needs to resample the spectrum.
    Parameters
    ----------
    in_file_name : str or file-like object
        path to a local file, or file-like object
    sim_type : str
        'vasp': .dat file from VASP simulation
        'crystal_out': .out file from CRYSTAL simulation, not preferred
        'crystal_dat': .dat file from CRYSTAL simulation
    use: one of 'I_tot', 'I_perp', 'I_par', 'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz'
        'I_tot', 'I_perp', 'I_par': available for both CRYSTAL and VASP
        'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz': available only for CRYSTAL
    """
    if isinstance(in_file, TextIOBase):
        labels, x, ydict = read_simulated_lines(in_file, sim_type=sim_type, use={use})
    else:
        with open(in_file) as f:
            labels, x, ydict = read_simulated_lines(f, sim_type=sim_type, use={use})
    y = ydict[use]
    spe = Spectrum(x=np.array(x), y=np.array(y))
    spe._sort_x()
    return spe
