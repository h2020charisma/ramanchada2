#!/usr/bin/env python3

from __future__ import annotations

import logging

from ..spectrum import Spectrum
from ramanchada.misc.spectrum_constructor import spectrum_constructor_deco

logger = logging.getLogger(__name__)


@spectrum_constructor_deco
def from_cache_or_calc(spe: Spectrum, requred_steps={}):
    try:
        spe.read_h5(repr(spe))
    except Exception as e:
        logger.warn(e)
