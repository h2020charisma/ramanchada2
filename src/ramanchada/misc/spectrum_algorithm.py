#!/usr/bin/env python3

from copy import copy
from ramanchada.spectrum.spectrum import Spectrum


def spectrum_algorithm_deco(fun):
    def retf(old_spe, *args, **kwargs):
        new_spe = copy(old_spe)
        fun(old_spe, new_spe, *args, **kwargs)
        new_spe._origin.append((fun.__name__, args, kwargs))
        new_spe.commit()
        return new_spe
    if hasattr(Spectrum, fun.__name__):
        raise ValueError(f'redefining {fun.__name__}')
    Spectrum._available_processings.add(fun.__name__)
    setattr(Spectrum, fun.__name__, retf)
    return retf
