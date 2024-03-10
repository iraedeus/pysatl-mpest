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
            func: Callable[[Params], float],
            params: Params,
            jacobian: Callable[[Params], np.ndarray]
    ) -> Params:
        pass


class ScipyNewtonCG(Optimizer):
    @staticmethod
    def name():
        return "ScipyNewtonCG"

    @staticmethod
    def minimize(
            func: Callable[[Params], float],
            params: Params,
            jacobian: Callable[[Params], np.ndarray]
    ) -> Params:
        return minimize(func, params, jacobian).x
