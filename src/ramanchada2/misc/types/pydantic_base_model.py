from abc import ABC, abstractmethod
from pydantic import BaseModel


class PydBaseModel(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def serialize(self):
        pass
