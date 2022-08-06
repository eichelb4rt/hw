from __future__ import annotations
from abc import ABC, abstractmethod


class Recommender(ABC):
    name: str

    @abstractmethod
    def fit(self, x_train) -> Recommender:
        pass

    @abstractmethod
    def rate(self, x_qualify):
        pass
