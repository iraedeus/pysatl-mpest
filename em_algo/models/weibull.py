"""TODO"""

import numpy as np
from scipy.stats import weibull_min

from em_algo.types import Samples, Params
from em_algo.models import AModelDifferentiable


class WeibullModelExp(AModelDifferentiable):
    """
    f(x) = (k / l) * (x / l)^(k - 1) / e^((x / l)^k)

    k = e^(_k)

    l = e^(_l)

    O = [_k, _l]
    """

    @staticmethod
    def name() -> str:
        return "WeibullExp"

    @staticmethod
    def params_convert_to_model(params: Params) -> Params:
        return np.log(params)

    @staticmethod
    def params_convert_from_model(params: Params) -> Params:
        return np.exp(params)

    @staticmethod
    def generate(params: Params, size: int = 1) -> Samples:
        return np.array(weibull_min.rvs(params[0], loc=0, scale=params[1], size=size))

    @staticmethod
    def p(x: float, params: Params) -> float:
        if x < 0:
            return 0
        ek, el = np.exp(params)
        xl = x / el
        return (ek / el) * (xl ** (ek - 1.0)) / np.exp(xl**ek)

    @staticmethod
    def lp(x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        k, l = params
        ek, el = np.exp(params)
        lx = np.log(x)
        return k - ((x / el) ** ek) - ek * l - lx + ek * lx

    @staticmethod
    def ldk(x: float, params: Params) -> float:
        """TODO"""

        if x < 0:
            return -np.inf
        ek, el = np.exp(params)
        xl = x / el
        return 1.0 - ek * ((xl**ek) - 1.0) * np.log(xl)

    @staticmethod
    def ldl(x: float, params: Params) -> float:
        """TODO"""

        if x < 0:
            return -np.inf
        ek, el = np.exp(params)
        return ek * ((x / el) ** ek - 1.0)

    @staticmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        return np.array(
            [WeibullModelExp.ldk(x, params), WeibullModelExp.ldl(x, params)]
        )
