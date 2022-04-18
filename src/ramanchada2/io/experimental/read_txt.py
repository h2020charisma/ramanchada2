#!/usr/bin/env python3

from typing import TextIO

from .bw_format import bw_format


""" There are 4 types of TXT data files that can be distinguished by their first line:
    1. File Version;BWRam4.11_11
    2. File Version;BWSpec4.11_1
    3. <wavenumber>	<intensity>
    4. #Wave		#Intensity
      <wavenumber>	<intensity>
    """


def read_txt(data_in: TextIO):
    lines = data_in.readlines()
    if lines[0].startswith('File Version;BW'):
        return bw_format(lines)
    else:
        raise NotImplementedError('filetype not supported, only BW filetype implemented')
