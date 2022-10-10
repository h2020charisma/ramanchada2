#!/usr/bin/env python3

from typing import TextIO, Tuple, Dict

from numpy.typing import NDArray
import numpy as np


def read_csv(data_in: TextIO) -> Tuple[NDArray, NDArray, Dict]:
    lines = data_in.readlines()
    positions, intensities = np.genfromtxt(lines, delimiter=',', dtype=float).T
    filter_nan = ~(
        np.isnan(positions) |
        np.isnan(intensities)
    )
    positions = positions[filter_nan]
    intensities = intensities[filter_nan]
    meta: Dict[str, None] = dict()
    return positions, intensities, meta
