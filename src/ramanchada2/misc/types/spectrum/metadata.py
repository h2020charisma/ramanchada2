#!/usr/bin/env python3

import datetime
import json
from typing import Any, Dict, List, Union

import numpy as np
import numpy.typing as npt
import pydantic

from ..pydantic_base_model import PydBaseModel

SpeMetadataFieldTyping = Union[
    npt.NDArray, PydBaseModel,
    pydantic.StrictBool,
    pydantic.StrictInt, float,
    datetime.datetime,
    List[Any], Dict[str, Any],
    pydantic.StrictStr]

SpeMetadataTyping = Dict[str, SpeMetadataFieldTyping]


class SpeMetadataFieldModel(PydBaseModel):
    __root__: SpeMetadataFieldTyping

    @pydantic.validator('__root__', pre=True)
    def pre_validate(cls, val):
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, str):
            if val.startswith('ramanchada2_model@'):
                # The format is:
                # ramanchada2_model@ModelName#<DATA>
                pos_at = val.index('@')
                pos_hash = val.index('#')
                model_name = val[pos_at+1:pos_hash]
                from ramanchada2.misc import types
                model = getattr(types, model_name)
                return model.validate(val[pos_hash+1:])
            if (val.startswith('[') and val.endswith(']') or
               val.startswith('{') and val.endswith('}')):
                return json.loads(val.replace("'", '"').replace(r'b"', '"'))
        return val

    def serialize(self):
        if isinstance(self.__root__, list) or isinstance(self.__root__, dict):
            return json.dumps(self.__root__)
        if isinstance(self.__root__, PydBaseModel):
            return f'ramanchada2_model@{type(self.__root__).__name__}#' + self.json()
        if isinstance(self.__root__, datetime.datetime):
            return self.__root__.isoformat()
        if isinstance(self.__root__, PydBaseModel):
            return self.__root__.serialize()
        return self.__root__


class SpeMetadataModel(PydBaseModel):
    __root__: Dict[str, SpeMetadataFieldModel]

    def __str__(self):
        return str(self.serialize())

    def serialize(self):
        return {k: v.serialize() for k, v in sorted(self.__root__.items())}

    def __getitem__(self, key: str) -> SpeMetadataFieldTyping:
        return self.__root__[key].__root__

    def _update(self, val: Dict):
        self.__root__.update(self.validate(val).__root__)

    def _del_key(self, key: str):
        del self.__root__[key]

    def _flush(self):
        self.__root__ = {}
