#!/usr/bin/env python3

from copy import copy
from ramanchada2.spectrum.spectrum import Spectrum


class spectrum_arithmetics_deco:
    def __init__(self, *args):
        self.arith = args

    def __call__(self, fun):
        for arith in self.arith:
            ar_fun = getattr(Spectrum.y, arith)

            def retf(old_spe, arg):
                new_spe = copy(old_spe)
                fun(old_spe, new_spe, ar_fun, arg)
                return new_spe
            if hasattr(Spectrum, arith):
                raise ValueError(f'redefining {arith}')
            setattr(Spectrum, arith, retf)
