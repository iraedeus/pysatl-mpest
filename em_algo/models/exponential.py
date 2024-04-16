"""TODO"""

import numpy as np
from scipy.stats import expon

from em_algo.types import Samples, Params
from em_algo.models import AModel


class ExponentialModel(AModel):
    """
    f(x) = l * e^(-lx)

    l = e^(_l)

    O = [_l]
    """

    @staticmethod
    def name() -> str:
        return "Exponential"

    @staticmethod
    def params_convert_to_model(params: Params) -> Params:
        return np.log(params)

    @staticmethod
    def params_convert_from_model(params: Params) -> Params:
        return np.exp(params)

    @staticmethod
    def generate(params: Params, size: int = 1) -> Samples:
        return np.array(expon.rvs(scale=1 / params[0], size=size))

    @staticmethod
    def p(x: float, params: Params) -> float:
        if x < 0:
            return 0
        (l,) = params
        return np.exp(l - np.exp(l) * x)

    @staticmethod
    def lp(x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        (l,) = params
        return l - np.exp(l) * x

    @staticmethod
    def ldl(x: float, params: Params) -> float:
        """TODO"""

        if x < 0:
            return -np.inf
        (l,) = params
        return 1 - np.exp(l) * x

    @staticmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        return np.array([ExponentialModel.ldl(x, params)])
