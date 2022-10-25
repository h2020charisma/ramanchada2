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
              x: npt.NDArray, y: npt.NDArray, meta: Dict, h5module=None):
    data = np.stack([x, y])
    try:
        _h5 = h5module or h5py
        with _h5.File(filename, mode='a') as h5:
            if h5.get(dataset) is None:
                ds = h5.create_dataset(dataset, data=data)
                ds.attrs.update(meta)
            else:
                logger.warning(f'dataset `{dataset}` already exists in file `{filename}`')
    except ValueError as e:
        logger.warning(repr(e))


def read_cha(filename: str,
             dataset: str, h5module=None
             ) -> Tuple[npt.NDArray, npt.NDArray, Dict]:
    _h5 = h5module or h5py
    with _h5.File(filename, mode='r') as h5:
        data = h5.get(dataset)
        if data is None:
            raise ChadaReadNotFoundError(f'dataset `{dataset}` not found in file `{filename}`')
        x, y = data[:]
        meta = dict(data.attrs)
    return x, y, meta


def filter_dataset(topdomain, domain, process_file, sample=None, wavelength=None, instrument=None,
                   provider=None, investigation=None, kwargs={}, h5module=None):
    _h5 = h5module or h5py
    with _h5.File(domain) as dataset:
        if (sample is not None) and (dataset["annotation_sample"].attrs["sample"] == sample):
            process_file(topdomain, domain, **kwargs)


def visit_domain(topdomain="/", process_dataset=None, kwargs={}, h5module=None):
    _h5 = h5module or h5py
    if topdomain.endswith("/"):
        with _h5.Folder(topdomain) as domain:
            domain._getSubdomains()
            for domain in domain._subdomains:
                if domain["class"] == "folder":
                    visit_domain("{}/".format(domain["name"]), process_dataset, kwargs, h5module=_h5)
                else:
                    if not (process_dataset is None):
                        process_dataset(topdomain, domain["name"], **kwargs, h5module=_h5)
    else:
        if not (process_dataset is None):
            process_dataset(None, topdomain, **kwargs, h5module=_h5)
