from typing import Literal

import numpy as np
from pydantic import validate_arguments

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ..spectrum import Spectrum
from .algos import first_derivative, gg_1spike, gg_2spike

METHODS = {
    'gg_1spike': gg_1spike,
    'gg_2spike': gg_2spike,
    'first_derivative': first_derivative
}


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_metric(spe: Spectrum, /,
                  method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                  ):
    return METHODS[method].metric(spe.y)


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_indices(spe: Spectrum, /,
                   method: Literal[tuple(METHODS.keys())],  # type: ignore [valid-type]
                   threshold=None):
    return METHODS[method].indices(spe.y, threshold=threshold)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_drop(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method,
                threshold=None):
    idx = METHODS[method].indices(old_spe.y, threshold=threshold)
    new_spe.x = np.delete(old_spe.x, idx)
    new_spe.y = np.delete(old_spe.y, idx)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_linfix(old_spe: Spectrum,
                  new_spe: Spectrum, /,
                  method,
                  threshold=None):
    idx = METHODS[method].indices(old_spe.y, threshold=threshold)
    x = np.delete(old_spe.x, idx)
    y = np.delete(old_spe.y, idx)
    new_spe.y = np.interp(old_spe.x, x, y)


@add_spectrum_filter
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def spikes_only(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method,
                threshold=None):
    idx = METHODS[method].indices(old_spe.y, threshold=threshold)
    x = np.delete(old_spe.x, idx)
    y = np.delete(old_spe.y, idx)
    new_spe.y = old_spe.y - np.interp(old_spe.x, x, y)
