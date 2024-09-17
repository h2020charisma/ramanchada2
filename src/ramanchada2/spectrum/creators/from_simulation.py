"""Create spectrum from simulation output files."""

from io import TextIOBase
import numpy as np
from typing import Dict, Literal, Union

from pydantic import PositiveFloat, PositiveInt, validate_call

from ramanchada2.io.simulated import read_simulated_lines
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor

from .from_delta_lines import from_delta_lines

_DIRECTION_LITERALS = Literal['I_tot', 'I_perp', 'I_par', 'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz']


@add_spectrum_constructor()
@validate_call(config=dict(arbitrary_types_allowed=True))
def from_simulation(in_file: Union[str, TextIOBase],
                    sim_type: Literal['vasp', 'crystal_out', 'crystal_dat', 'raw_dat'],
                    use: Union[_DIRECTION_LITERALS, Dict[_DIRECTION_LITERALS, PositiveFloat]] = 'I_tot',
                    nbins: PositiveInt = 2000,
                    ):
    """
    Generate spectrum from simulation file.

    The returned spectrum has only few x/y pairs -- one for each simulated line. Values along
    the x-axis will not be uniform. To make it uniform, one needs to resample the spectrum.

    Args:
        in_file:
            Path to a local file, or file-like object.
        sim_type:
            If `vasp`: `.dat` file from VASP simulation. If `crystal_out`: `.out` file from CRYSTAL simulation, not
            preferred. If `crystal_dat`: `.dat` file from CRYSTAL simulation.
        use:
            One of the directions `I_tot`, `I_perp`, `I_par`, `I_xx`, `I_xy`,
            `I_xz`, `I_yy`, `I_yz`, `I_zz`, `I_tot`, `I_perp`, `I_par` are
            available for both CRYSTAL and VASP. `I_xx`, `I_xy`, `I_xz`,
            `I_yy`, `I_yz`, `I_zz` are available only for CRYSTAL. If a Dict is
            passed, the key should be directions and values should be weighting factor.
            For example, `use={'I_perp': .1, 'I_par': .9}`

    """
    if isinstance(use, str):
        use_directions = {use}
    else:
        use_directions = set(use.keys())
    if isinstance(in_file, TextIOBase):
        labels, x, ydict = read_simulated_lines(in_file, sim_type=sim_type, use=use_directions)
    else:
        with open(in_file) as f:
            labels, x, ydict = read_simulated_lines(f, sim_type=sim_type, use=use_directions)
    if isinstance(use, str):
        y = ydict[use]
    else:
        dirs = list(use.keys())
        fact = np.array(list(use.values()))
        y = np.transpose([ydict[i] for i in dirs]) @ fact
    spe = from_delta_lines(deltas=dict(zip(x, y)), nbins=nbins)
    return spe
