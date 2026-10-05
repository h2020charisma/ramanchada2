import datetime
import json
from typing import Any, Dict, List, Union

import numpy as np
import numpy.typing as npt
from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator

from ..pydantic_base_model import PydBaseModel, PydRootModel

SpeMetadataFieldTyping = Union[
    npt.NDArray,
    StrictBool,
    StrictInt, float,
    datetime.datetime,
    List[Any], Dict[str, Any],
    StrictStr, None]


class SpeMetadataFieldModel(PydRootModel):
    root: SpeMetadataFieldTyping = Field(union_mode='left_to_right')

    @field_validator('root', mode='before')
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
                return model.model_validate(val[pos_hash+1:])
            if (val.startswith('[') and val.endswith(']') or
               val.startswith('{') and val.endswith('}')):
                # Not every "[...]"/"{...}"-bracketed string is JSON --
                # e.g. RRUFF chemistry-formula metadata legitimately uses
                # square brackets for coordination notation (confirmed:
                # "[Ca_2_(H_2_O)_17_Mg(H_2_...)]"), which raises
                # json.JSONDecodeError here instead of the "[1, 2, 3]"-style
                # list/dict values this branch actually targets. Fall back
                # to the plain string when it isn't real JSON, rather than
                # letting the decode error propagate as a validation
                # failure for every such value.
                try:
                    return json.loads(val.replace("'", '"').replace(r'b"', '"'))
                except json.JSONDecodeError:
                    return val
        return val

    def serialize(self):
        if isinstance(self.root, list) or isinstance(self.root, dict):
            return json.dumps(self.root)
        if isinstance(self.root, PydBaseModel):
            return f'ramanchada2_model@{type(self.root).__name__}#' + self.json()
        if isinstance(self.root, datetime.datetime):
            return self.root.isoformat()
        if isinstance(self.root, PydBaseModel):
            return self.root.serialize()
        return self.root


class SpeMetadataModel(PydRootModel):
    root: Dict[str, SpeMetadataFieldModel]

    @field_validator('root', mode='before')
    def pre_validate(cls, val):
        if val is None or val == '':
            val = {}
        elif isinstance(val, list):
            val = {'%d' % k: v for k, v in enumerate(val)}
        return val

    def __str__(self):
        return str(self.serialize())

    def serialize(self):
        return {k: v.serialize() for k, v in sorted(self.root.items())}

    def __getitem__(self, key: str) -> SpeMetadataFieldTyping:
        return self.root[key].root

    def _update(self, val: Dict):
        self.root.update(self.model_validate(val).root)

    def _del_key(self, key: str):
        del self.root[key]

    def _flush(self):
        self.root = {}

    def get_all_keys(self) -> list[str]:
        """
        Returns a list of all keys in the metadata model.
        """
        return list(self.root.keys())
