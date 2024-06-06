"""Module which contains Weibull model class"""

import numpy as np
from scipy.stats import weibull_min

from em_algo.types import Samples, Params
from em_algo.models import AModelDifferentiable, AModelWithGenerator


class WeibullModelExp(AModelDifferentiable, AModelWithGenerator):
    """
    f(x) = (k / l) * (x / l)^(k - 1) / e^((x / l)^k)

    k = e^(_k)

    l = e^(_l)

    O = [_k, _l]
    """

    @property
    def name(self) -> str:
        return "WeibullExp"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.log(params)

    def params_convert_from_model(self, params: Params) -> Params:
        return np.exp(params)

    def generate(self, params: Params, size: int = 1) -> Samples:
        return np.array(weibull_min.rvs(params[0], loc=0, scale=params[1], size=size))

    def pdf(self, x: float, params: Params) -> float:
        if x < 0:
            return 0
        ek, el = np.exp(params)
        xl = x / el
        return (ek / el) * (xl ** (ek - 1.0)) / np.exp(xl**ek)

    def lpdf(self, x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        k, l = params
        ek, el = np.exp(params)
        lx = np.log(x)
        return k - ((x / el) ** ek) - ek * l - lx + ek * lx

    def ldk(self, x: float, params: Params) -> float:
        """Method which returns logarithm of derivative with respect to k"""

        if x < 0:
            return -np.inf
        ek, el = np.exp(params)
        xl = x / el
        return 1.0 - ek * ((xl**ek) - 1.0) * np.log(xl)

    def ldl(self, x: float, params: Params) -> float:
        """Method which returns logarithm of derivative with respect to l"""

        if x < 0:
            return -np.inf
        ek, el = np.exp(params)
        return ek * ((x / el) ** ek - 1.0)

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        return np.array([self.ldk(x, params), self.ldl(x, params)])
