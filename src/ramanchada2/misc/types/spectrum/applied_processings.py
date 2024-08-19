from __future__ import annotations

from typing import Dict, List

from pydantic import (BaseModel, Field, RootModel, field_validator,
                      validate_call)


class SpeProcessingModel(BaseModel):
    proc: str = Field(...)
    args: List = list()
    kwargs: Dict = dict()

    @property
    def is_constructor(self):
        from ramanchada2.misc.spectrum_deco.dynamically_added import \
            dynamically_added_constructors
        return self.proc in dynamically_added_constructors

    @field_validator('proc', mode='before')
    @validate_call
    def check_proc(cls, val: str):
        from ramanchada2.misc.spectrum_deco.dynamically_added import (
            dynamically_added_constructors, dynamically_added_filters)
        if val in dynamically_added_filters:
            return val
        if val in dynamically_added_constructors:
            return val
        raise ValueError(f'processing {val} not supported')


class SpeProcessingListModel(RootModel):
    root: List[SpeProcessingModel]

    def __len__(self):
        return len(self.root)

    def append(self, proc, args=[], kwargs={}):
        self.root.append(SpeProcessingModel(proc=proc, args=args, kwargs=kwargs))

    @validate_call
    def extend_left(self, proc_list: List[SpeProcessingModel]):
        self.root = proc_list + self.root

    def pop(self):
        return self.root.pop()

    def clear(self):
        return self.root.clear()

    def assign(self, *args, **kwargs):
        self.clear()
        self.append(*args, **kwargs)

    def _string_list(self):
        ret = list()
        for elem in self.root:
            args = [f'{repr(i)}' for i in elem.args]
            kwargs = [f'{k}={repr(v)}' for k, v in elem.kwargs.items()]
            comb = ', '.join(args + kwargs)
            ret.append(f'{elem.proc}({comb})')
        return ret

    def repr(self):
        return '.'.join(self._string_list())

    def cache_path(self):
        return '/'.join([
            i.replace(' ', '').replace('/', '_')
            for i in self._string_list()])

    def to_list(self):
        return [i.model_dump() for i in self.root]
