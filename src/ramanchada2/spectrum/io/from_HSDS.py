#!/usr/bin/env python3

from __future__ import annotations

import h5py
import logging

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_method import spectrum_method_deco
from ramanchada2.misc.spectrum_constructor import spectrum_constructor_deco

logger = logging.getLogger()


@spectrum_method_deco
def write_h5(spe: Spectrum):
    filename = f'{spe.cachedir}/{spe.h5file}'
    base_dataset = f'{repr(spe)}/_Spectrum'.replace(' ', '')
    with h5py.File(filename, 'a') as h5:
        if h5.get(base_dataset):
            logger.warning(f'{base_dataset} already present in {filename}')
        else:
            logger.info(f'writing {base_dataset} to {filename}')
            h5.create_dataset(base_dataset + '/x', data=spe.x)
            h5.create_dataset(base_dataset + '/y', data=spe.y)


@spectrum_constructor_deco
def read_h5(spe: Spectrum, filename):
    ...
