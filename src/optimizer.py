from abc import ABC, abstractmethod
from scipy.optimize import minimize
from typing import Callable
from utils import *


class Optimizer(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def minimize(
            func: Callable[[params], float],
            params: params,
            jacobian: Callable[[params], np.ndarray]
    ) -> params:
        pass


class ScipyNewtonCG(Optimizer):
    @staticmethod
    def name():
        return "ScipyNewtonCG"

    @staticmethod
    def minimize(
            func: Callable[[params], float],
            params: params,
            jacobian: Callable[[params], np.ndarray]
    ) -> params:
        return minimize(func, params, jac=jacobian, method="Newton-CG").x
