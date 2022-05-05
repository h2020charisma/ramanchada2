#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def two_column_spe(lines: List[str]) -> Tuple[NDArray, NDArray]:
    """
    Parse two column spectrum.

    Parameters
    ----------
    lines : List[str]
        list of lines to parse

    Returns
    -------
    Tuple[NDArray, NDArray]
        - positions
        - intensities
    """
    positions, intensities = np.genfromtxt(lines).T
    return positions, intensities
