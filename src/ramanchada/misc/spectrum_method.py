#!/usr/bin/env python

from ramanchada.spectrum.spectrum import Spectrum


def spectrum_method_deco(fun):
    def retf(spe: Spectrum, *args, **kwargs):
        fun(spe, *args, **kwargs)
    if hasattr(Spectrum, fun.__name__):
        raise ValueError(f'redefining {fun.__name__}')
    setattr(Spectrum, fun.__name__, retf)
    return retf
