#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, List

import pydantic


class SpeProcessingModel(pydantic.BaseModel):
    proc: str = pydantic.Field(...)
    args: List = list()
    kwargs: Dict = dict()

    @property
    def is_constructor(self):
        from ramanchada2.misc.spectrum_deco.dynamically_added import dynamically_added_constructors
        return self.proc in dynamically_added_constructors

    @pydantic.validator('proc', pre=True)
    @pydantic.validate_arguments
    def check_proc(cls, val: str):
        from ramanchada2.misc.spectrum_deco.dynamically_added import (dynamically_added_filters,
                                                                      dynamically_added_constructors)
        if val in dynamically_added_filters:
            return val
        if val in dynamically_added_constructors:
            return val
        raise ValueError(f'processing {val} not supported')


class SpeProcessingListModel(pydantic.BaseModel):
    __root__: List[SpeProcessingModel]

    def __len__(self):
        return len(self.__root__)

    def append(self, proc, args=[], kwargs={}):
        self.__root__.append(SpeProcessingModel(proc=proc, args=args, kwargs=kwargs))

    @pydantic.validate_arguments
    def extend_left(self, proc_list: List[SpeProcessingModel]):
        self.__root__ = proc_list + self.__root__

    def pop(self):
        return self.__root__.pop()

    def clear(self):
        return self.__root__.clear()

    def assign(self, *args, **kwargs):
        self.clear()
        self.append(*args, **kwargs)

    def _string_list(self):
        ret = list()
        for elem in self.__root__:
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
        return [i.dict() for i in self.__root__]
