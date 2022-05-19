#!/usr/bin/env python3

from functools import wraps
from ramanchada2.spectrum.spectrum import Spectrum
from .dynamically_added import dynamically_added_constructors


class add_spectrum_constructor:
    def __init__(self, set_applied_processing=True):
        self.set_proc = set_applied_processing

    def __call__(self, fun):
        @wraps(fun)
        def retf(*args, cachefile_=None, **kwargs) -> Spectrum:
            spe = fun(*args, **kwargs)
            if self.set_proc:
                spe._applied_processings.assign(proc=fun.__name__, args=args, kwargs=kwargs)
                if cachefile_:
                    spe._cachefile = cachefile_
                spe.write_cache()
            return spe
        if hasattr(Spectrum, fun.__name__):
            raise ValueError(f'redefining {fun.__name__}')
        setattr(Spectrum, fun.__name__, retf)
        dynamically_added_constructors.add(fun.__name__)
        return retf
