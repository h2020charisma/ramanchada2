import numpy as np

from renishawWiRE import WDFReader
from spc_io import SPC
from brukeropusreader import read_file


def readWDF(file):
    s = WDFReader(file)
    y_data = s.spectra
    x_data = s.xdata
    if np.mean(np.diff(x_data)) < 0:
        y_data = np.flip(y_data)
        x_data = np.flip(x_data)
    static_metadata = {
        "laser wavelength": s.laser_length,
        "no. of accumulations": s.accumulation_count,
        "spectral unit": s.spectral_unit.name,
        "OEM software name": s.application_name,
        "OEM software version": s.application_version
        }
    return x_data, y_data, static_metadata


def readSPC(file):
    with open(file, 'rb') as f:
        spc = SPC.from_bytes_io(f)
    if len(spc) != 1:
        raise ValueError(f'Only 1 sub supported, {len(spc)} found')
    x_data = spc[0].xarray
    y_data = spc[0].yarray
    static_metadata = spc.log_book.text
    return x_data, y_data, static_metadata


def readOPUS(file, obj_no=0):
    opus_data = read_file(file)
    x = opus_data.get_range("AB")
    y = opus_data["AB"]
    meta = {}
    for key in opus_data:
        if key == "AB":
            continue
        if isinstance(opus_data[key], dict):
            for subkey in opus_data[key]:
                meta["{}.{}".format(key, subkey)] = opus_data[key][subkey]
        else:
            meta[key] = opus_data[key]
    return x, y, meta
