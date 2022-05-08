#!/usr/bin/env python3

from typing import Dict, Union, Any, List
import datetime
import json

import pydantic
import numpy as np
import numpy.typing as npt


SpeMetadataFieldTyping = Union[
    npt.NDArray, pydantic.StrictBool,
    pydantic.StrictInt, float,
    datetime.datetime,
    List[Any], Dict[str, Any],
    pydantic.StrictStr]

SpeMetadataTyping = Dict[str, SpeMetadataFieldTyping]


class SpeMetaBaseModel(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True


class SpeMetadataFieldModel(SpeMetaBaseModel):
    __root__: SpeMetadataFieldTyping

    @pydantic.validator('__root__', pre=True)
    def pre_validate(cls, val):
        if isinstance(val, np.ndarray):
            return val
        elif (isinstance(val, str) and (
            val.startswith('[') and val.endswith(']') or
                val.startswith('{') and val.endswith('}'))):
            return json.loads(val.replace("'", '"'))
        else:
            return val


class SpeMetadataModel(SpeMetaBaseModel):
    __root__: Dict[str, SpeMetadataFieldModel]

    def __str__(self):
        return str(self.serialize())

    def serialize(self):
        ret = dict()
        for key, val in self.__root__.items():
            if isinstance(val.__root__, list) or isinstance(val.__root__, dict):
                ret.update({key: json.dumps(val.__root__)})
            elif isinstance(val.__root__, datetime.datetime):
                ret.update({key: val.__root__.isoformat()})
            else:
                ret.update({key: val.__root__})
        return ret

    def __getitem__(self, key: str) -> SpeMetadataFieldTyping:
        return self.__root__[key].__root__

    def _update(self, val: Dict):
        self.__root__.update(self.validate(val).__root__)

    def _del_key(self, key: str):
        del self.__root__[key]

    def _flush(self):
        self.__root__ = {}
