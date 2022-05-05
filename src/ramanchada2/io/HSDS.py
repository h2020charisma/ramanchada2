#!/usr/bin/env python3

import h5py
import logging

from ..spectrum import Spectrum

logger = logging.getLogger()


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


def read_h5(spe: Spectrum, filename):
    ...
