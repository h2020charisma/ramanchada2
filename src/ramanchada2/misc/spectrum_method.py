#!/usr/bin/env python

from ramanchada2.spectrum.spectrum import Spectrum


def spectrum_method_deco(fun):
    def retf(spe: Spectrum, *args, **kwargs):
        fun(spe, *args, **kwargs)
    if hasattr(Spectrum, fun.__name__):
        raise ValueError(f'redefining {fun.__name__}')
    setattr(Spectrum, fun.__name__, retf)
    retf.__name__ = fun.__name__
    retf.__doc__ = fun.__doc__
    return retf
