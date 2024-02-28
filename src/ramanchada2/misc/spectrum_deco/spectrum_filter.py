#!/usr/bin/env python3

from functools import wraps
from copy import copy, deepcopy
from ramanchada2.spectrum.spectrum import Spectrum
from .dynamically_added import dynamically_added_filters
import logging

logger = logging.getLogger(__name__)


def add_spectrum_filter(fun):
    @wraps(fun)
    def retf(old_spe: Spectrum, *args, **kwargs) -> Spectrum:
        new_spe = copy(old_spe)
        new_spe._applied_processings.append(proc=fun.__name__,
                                            args=deepcopy(args),
                                            kwargs=deepcopy(kwargs))
        fun(old_spe, new_spe, *args, **kwargs)
        new_spe.write_cache()
        return new_spe
    if hasattr(Spectrum, fun.__name__):
        raise ValueError(f'redefining {fun.__name__}')
    Spectrum._available_processings.add(fun.__name__)
    setattr(Spectrum, fun.__name__, retf)
    dynamically_added_filters.add(fun.__name__)
    return retf
