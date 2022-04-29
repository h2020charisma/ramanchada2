#!/usr/bin/env python3

from ramanchada2.spectrum.spectrum import Spectrum


def spectrum_constructor_deco(fun):
    def retf(*args, **kwargs) -> Spectrum:
        if 'spe_kwargs' in kwargs:
            spe_kwargs = kwargs['spe_kwargs']
            kwargs.pop('spe_kwargs')
        else:
            spe_kwargs = dict()
        spe = Spectrum(**spe_kwargs)
        fun(spe, *args, **kwargs)
        spe._origin.append((fun.__name__, args, kwargs))
        spe.commit()
        return spe
    if hasattr(Spectrum, fun.__name__):
        raise ValueError(f'redefining {fun.__name__}')
    setattr(Spectrum, fun.__name__, retf)
    return retf
