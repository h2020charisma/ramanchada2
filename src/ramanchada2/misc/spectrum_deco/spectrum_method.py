#!/usr/bin/env python

from functools import wraps
from ramanchada2.spectrum.spectrum import Spectrum


def spectrum_method_deco(fun):
    @wraps(fun)
    def retf(spe: Spectrum, *args, **kwargs):
        ret = fun(spe, *args, **kwargs)
        return ret
    if hasattr(Spectrum, fun.__name__):
        raise ValueError(f'redefining {fun.__name__}')
    setattr(Spectrum, fun.__name__, retf)
    return retf
