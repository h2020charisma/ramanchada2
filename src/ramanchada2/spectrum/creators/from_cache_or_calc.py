#!/usr/bin/env python3

from __future__ import annotations

import logging
from typing import Callable, List, Dict, Optional
from pydantic import BaseModel, Field, validator, validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import spectrum_constructor_deco

logger = logging.getLogger(__name__)


@spectrum_constructor_deco
def from_cache_or_calc(spe: Spectrum, requred_steps={}):
    try:
        # spe.read_h5(repr(spe))
        pass
    except Exception as e:
        logger.warn(e)


class Processing(BaseModel):
    proc: Callable = Field(...)
    args: Optional[List] = []
    kwargs: Optional[Dict] = dict()

    @validator('proc', pre=True)
    @validate_arguments
    def check_proc(cls, val: str):
        if not hasattr(Spectrum, val):
            print(repr(val))
            raise ValueError(f'processing {val} not supported')
        return getattr(Spectrum, val)


class ProcessingList(BaseModel):
    procs: List[Processing]
