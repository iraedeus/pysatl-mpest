import numpy as np
from scipy.stats import pareto

from mpest import Params, Samples
from mpest.models import AModelDifferentiable, AModelWithGenerator


class Pareto(AModelWithGenerator, AModelDifferentiable):
    @property
    def name(self) -> str:
        return "Pareto"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.log(params)

    def params_convert_from_model(self, params: Params) -> Params:
        return np.exp(params)

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        if not normalized:
            x0, k = params
            return pareto.rvs(b=k, scale=x0, size=size)

        x0, k = self.params_convert_from_model(params)
        return pareto.rvs(b=k, scale=x0, size=size)

    def pdf(self, x: float, params: Params) -> float:
        theta_x0, theta_k = params
        e_x0 = np.exp(theta_x0)
        if x < e_x0:
            return 0.0

        e_k = np.exp(theta_k)
        num = e_k * np.exp(theta_x0 * e_k)
        denom = x ** (e_k + 1)

        return num / denom

    def lpdf(self, x: float, params: Params) -> float:
        theta_x0, theta_k = params
        e_x0, e_k = np.exp(params)
        log = np.log(x)

        if x < e_x0:
            return -np.inf

        return theta_k + e_k * (theta_x0 - log) - log

    def ld_theta_x0(self, x: float, params: Params) -> float:
        e_x0, e_k = np.exp(params)

        if x < e_x0:
            return -np.inf

        return e_k

    def ld_theta_k(self, x: float, params: Params) -> float:
        theta_x0, theta_k = params
        e_x0 = np.exp(params[0])

        if x < e_x0:
            return -np.inf

        return 1 + np.exp(theta_k) * (theta_x0 - np.log(x))

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        return np.array([self.ld_theta_x0(x, params), self.ld_theta_k(x, params)])
