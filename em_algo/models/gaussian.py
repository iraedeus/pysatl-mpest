"""TODO"""

import numpy as np
from scipy.stats import norm

from em_algo.types import Samples, Params
from em_algo.models import AModelDifferentiable


class GaussianModel(AModelDifferentiable):
    """
    f(x) = e^(-1/2 * ((x - m) / sd)^2) / (sd * sqrt(2pi))

    sd = e^(_sd)

    O = [m, _sd]
    """

    @staticmethod
    def name() -> str:
        return "Gaussian"

    @staticmethod
    def params_convert_to_model(params: Params) -> Params:
        return np.array([params[0], np.log(params[1])])

    @staticmethod
    def params_convert_from_model(params: Params) -> Params:
        return np.array([params[0], np.exp(params[1])])

    @staticmethod
    def generate(params: Params, size: int = 1) -> Samples:
        return np.array(norm.rvs(loc=params[0], scale=params[1], size=size))

    @staticmethod
    def p(x: float, params: Params) -> float:
        m, sd = params
        sd = np.exp(sd)
        return np.exp(-0.5 * (((x - m) / sd) ** 2)) / (sd * np.sqrt(2 * np.pi))

    @staticmethod
    def lp(x: float, params: Params) -> float:
        p = GaussianModel.p(x, params)
        if p <= 0:
            return -np.inf
        return np.log(p)

    @staticmethod
    def ldm(x: float, params: Params) -> float:
        """TODO"""

        m, sd = params
        return (x - m) / (np.exp(2 * sd))

    @staticmethod
    def ldsd(x: float, params: Params) -> float:
        """TODO"""

        m, sd = params
        return ((x - m) ** 2) / np.exp(2 * sd) - 1

    @staticmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        return np.array([GaussianModel.ldm(x, params), GaussianModel.ldsd(x, params)])
