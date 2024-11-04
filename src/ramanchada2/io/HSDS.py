import logging
from typing import Dict, Tuple

import h5py
import numpy as np
import numpy.typing as npt
from pydantic import validate_call

from ramanchada2.misc.exceptions import ChadaReadNotFoundError

logger = logging.getLogger()


# https://manual.nexusformat.org/examples/napi/python.html
# https://manual.nexusformat.org/examples/python/simple_example_basic/index.html
@validate_call(config=dict(arbitrary_types_allowed=True))
def write_nexus(filename: str,
                dataset: str,
                x: npt.NDArray, y: npt.NDArray, meta: Dict, h5module=None):
    _h5 = h5module or h5py
    try:
        with _h5.File(filename, 'a') as f:
            f.attrs['default'] = dataset
            try:
                nxentry = f.require_group('sample')
            except:  # noqa: E722
                pass

            nxentry = f.require_group('instrument')
            for m in meta:
                print(m, meta[m])

            try:
                nxentry = f.require_group(dataset)
                nxentry.attrs["NX_class"] = 'NXentry'
                nxentry.attrs['default'] = 'data'
            except:  # noqa: E722
                pass

            try:
                nxdata = nxentry.require_group('data')
                nxdata.attrs["NX_class"] = 'NXdata'
                nxdata.attrs['signal'] = 'spectrum'
                nxdata.attrs['axes'] = 'raman_shift'
                nxdata.attrs['raman_shift_indices'] = [0,]
            except:  # noqa: E722
                pass

            try:
                tth = nxdata.require_group('raman_shift', data=x)
                tth.attrs['units'] = 'cm-1'
                tth.attrs['long_name'] = 'Raman shift (cm-1)'
            except:  # noqa: E722
                pass

            try:
                counts = nxdata.create_dataset('spectrum', data=y)
                counts.attrs['units'] = 'au'
                counts.attrs['long_name'] = 'spectrum'
            except:  # noqa: E722
                pass

    except ValueError as e:
        logger.warning(repr(e))


class DatasetExistsError(Exception):
    pass


def sanitize_key(key: str) -> str:
    return ''.join(c if ord(c) < 128 else '_' for c in key)


@validate_call(config=dict(arbitrary_types_allowed=True))
def write_cha(filename: str,
              dataset: str,
              x: npt.NDArray, y: npt.NDArray, meta: Dict, h5module=None):
    data = np.stack([x, y])
    sanitized_meta = {sanitize_key(k): v for k, v in meta.items()}
    try:
        _h5 = h5module or h5py
        with _h5.File(filename, mode='a') as h5:
            if h5.get(dataset) is None:
                ds = h5.create_dataset(dataset, data=data)
                ds.attrs.update(sanitized_meta)

            else:
                raise DatasetExistsError(f'dataset `{dataset}` already exists in file `{filename}`')
    except ValueError as e:
        raise e


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
