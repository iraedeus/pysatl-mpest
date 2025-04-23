import numpy as np
from scipy.special import betaln, digamma, expm1
from scipy.stats import beta

from mpest import Params, Samples
from mpest.models import AModelDifferentiable, AModelWithGenerator


class Beta(AModelWithGenerator, AModelDifferentiable):
    @property
    def name(self) -> str:
        return "Beta"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.log(params)

    def params_convert_from_model(self, params: Params) -> Params:
        return np.exp(params)

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        if not normalized:
            return beta.rvs(a=params[0], b=params[1], size=size)

        a, b = self.params_convert_from_model(params)
        return beta.rvs(a=a, b=b, size=size)

    def pdf(self, x: float, params: Params) -> float:
        return np.exp(self.lpdf(x, params))

    def lpdf(self, x: float, params: Params) -> float:
        if not (0 < x < 1):
            return -np.inf

        mu, nu = params
        lbeta = -betaln(np.exp(params[0]), np.exp(params[1]))
        log1 = expm1(mu) * np.log(x)
        log2 = expm1(nu) * np.log(1 - x)

        return lbeta + log1 + log2

    def ldmu(self, x: float, params: Params) -> float:
        if not (0 < x < 1):
            return -np.inf

        a, b = self.params_convert_from_model(params)
        return a * (digamma(a + b) - digamma(a) + np.log(x))

    def ldnu(self, x: float, params: Params) -> float:
        if not (0 < x < 1):
            return -np.inf

        a, b = self.params_convert_from_model(params)
        return b * (digamma(a + b) - digamma(b) + np.log(1 - x))

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        return np.array([self.ldmu(x, params), self.ldnu(x, params)])
