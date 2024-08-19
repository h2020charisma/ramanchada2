from abc import ABC, abstractmethod

from pydantic import BaseModel, RootModel


class PydBaseModel(BaseModel, ABC):
    model_config = dict(
        arbitrary_types_allowed=True
    )

    @abstractmethod
    def serialize(self):
        pass


class PydRootModel(RootModel, ABC):
    model_config = dict(
        arbitrary_types_allowed=True
    )

    @abstractmethod
    def serialize(self):
        pass
