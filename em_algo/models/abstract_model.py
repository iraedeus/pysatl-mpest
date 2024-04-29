"""TODO"""

from abc import ABC, abstractmethod
import numpy as np

from em_algo.types import Samples, Params
from em_algo.utils import Named


class AModel(Named, ABC):
    """TODO"""

    @abstractmethod
    def params_convert_to_model(self, params: Params) -> Params:
        """TODO"""

    @abstractmethod
    def params_convert_from_model(self, params: Params) -> Params:
        """TODO"""

    @abstractmethod
    def generate(self, params: Params, size: int = 1) -> Samples:
        """TODO"""

    @abstractmethod
    def p(self, x: float, params: Params) -> float:
        """TODO"""

    @abstractmethod
    def lp(self, x: float, params: Params) -> float:
        """TODO"""


class AModelDifferentiable(AModel, ABC):
    """TODO"""

    @abstractmethod
    def ld_params(self, x: float, params: Params) -> np.ndarray:
        """TODO"""
