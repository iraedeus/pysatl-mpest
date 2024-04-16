"""TODO"""

from abc import ABC, abstractmethod
import numpy as np

from em_algo.types import Samples, Params


class Model(ABC):
    """TODO"""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """TODO"""

    @staticmethod
    @abstractmethod
    def params_convert_to_model(params: Params) -> Params:
        """TODO"""

    @staticmethod
    @abstractmethod
    def params_convert_from_model(params: Params) -> Params:
        """TODO"""

    @staticmethod
    @abstractmethod
    def generate(params: Params, size: int = 1) -> Samples:
        """TODO"""

    @staticmethod
    @abstractmethod
    def p(x: float, params: Params) -> float:
        """TODO"""

    @staticmethod
    @abstractmethod
    def lp(x: float, params: Params) -> float:
        """TODO"""

    @staticmethod
    @abstractmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        """TODO"""
