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

    @property
    def name(self) -> str:
        return "Gaussian"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.array([params[0], np.log(params[1])])

    def params_convert_from_model(self, params: Params) -> Params:
        return np.array([params[0], np.exp(params[1])])

    def generate(self, params: Params, size: int = 1) -> Samples:
        return np.array(norm.rvs(loc=params[0], scale=params[1], size=size))

    def pdf(self, x: float, params: Params) -> float:
        m, sd = params
        sd = np.exp(sd)
        return np.exp(-0.5 * (((x - m) / sd) ** 2)) / (sd * np.sqrt(2 * np.pi))

    def lpdf(self, x: float, params: Params) -> float:
        p = self.pdf(x, params)
        if p <= 0:
            return -np.inf
        return np.log(p)

    def ldm(self, x: float, params: Params) -> float:
        """TODO"""

        m, sd = params
        return (x - m) / (np.exp(2 * sd))

    def ldsd(self, x: float, params: Params) -> float:
        """TODO"""

        m, sd = params
        return ((x - m) ** 2) / np.exp(2 * sd) - 1

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        return np.array([self.ldm(x, params), self.ldsd(x, params)])
