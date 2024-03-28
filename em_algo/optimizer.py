"""TODO"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy.optimize import minimize

from em_algo.utils import Params


class Optimizer(ABC):
    """TODO"""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """TODO"""

    @staticmethod
    @abstractmethod
    def minimize(
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        """TODO"""


class ScipyNewtonCG(Optimizer):
    """TODO"""

    @staticmethod
    def name():
        return "ScipyNewtonCG"

    @staticmethod
    def minimize(
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        return minimize(func, params, jac=jacobian, method="Newton-CG").x
