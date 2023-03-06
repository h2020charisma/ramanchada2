import numpy as np

from renishawWiRE import WDFReader
from spc_spectra import spc
import opusFC


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
    s = spc.File(file)
    if len(s.sub) != 1:
        raise ValueError(f'Only 1 sub supported, {len(s.sub)} found')
    x_data = s.x
    y_data = s.sub[0].y
    static_metadata = s.log_dict
    return x_data, y_data, static_metadata


def readOPUS(file, obj_no=0):
    c = opusFC.listContents(file)
    data = opusFC.getOpusData(file, c[obj_no])
    return data.x, data.y, data.parameters
