#!/usr/bin/env python3

from typing import Tuple, Dict

import h5py
import logging
import numpy as np
import numpy.typing as npt
import pydantic

from ramanchada2.misc.exceptions import ChadaReadNotFoundError

logger = logging.getLogger()


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def write_cha(filename: str,
              dataset: str,
              x: npt.NDArray, y: npt.NDArray, meta: Dict):
    data = np.stack([x, y])
    try:
        with h5py.File(filename, 'a') as h5:
            if h5.get(dataset) is None:
                ds = h5.create_dataset(dataset, data=data)
                ds.attrs.update(meta)
            else:
                logger.warning(f'dataset `{dataset}` already exists in file `{filename}`')
    except ValueError as e:
        logger.warning(repr(e))


def read_cha(filename: str,
             dataset: str
             ) -> Tuple[npt.NDArray, npt.NDArray, Dict]:
    with h5py.File(filename, 'r') as h5:
        data = h5.get(dataset)
        if data is None:
            raise ChadaReadNotFoundError(f'dataset `{dataset}` not found in file `{filename}`')
        x, y = data[:]
        meta = dict(data.attrs)
    return x, y, meta
