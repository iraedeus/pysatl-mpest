from abc import ABC, abstractmethod
import numpy as np
from utils import *
from scipy.stats import weibull_min, norm


class Model(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def params_convert_to_model(O: params) -> params:
        pass

    @staticmethod
    @abstractmethod
    def params_convert_from_model(O: params) -> params:
        pass

    @staticmethod
    @abstractmethod
    def generate(O: params, size: int = 1) -> sample:
        pass

    @staticmethod
    @abstractmethod
    def p(x: float, O: params) -> float:
        pass

    @staticmethod
    @abstractmethod
    def lp(x: float, O: params) -> float:
        pass

    @staticmethod
    @abstractmethod
    def ldO(x: float, O: params) -> np.ndarray:
        pass


class WeibullModelExp(Model):
    """
    f(x) = (k / l) * (x / l)^(k - 1) / e^((x / l)^k)

    k = e^(_k)

    l = e^(_l)

    O = {_k, _l}
    """
    @staticmethod
    def name() -> str:
        return "WeibullExp"

    @staticmethod
    def params_convert_to_model(O: params) -> params:
        return np.log(O)

    @staticmethod
    def params_convert_from_model(O: params) -> params:
        return np.exp(O)

    @staticmethod
    def generate(O: params, size: int = 1) -> sample:
        return np.array(weibull_min.rvs(
            O[0],
            loc=0,
            scale=O[1],
            size=size
        ))

    @staticmethod
    def p(x: float, O: params) -> float:
        if x < 0:
            return 0
        ek, el = np.exp(O)
        xl = x / el
        return (ek / el) * (xl ** (ek - 1.0)) / np.exp(xl**ek)

    @staticmethod
    def lp(x: float, O: params) -> float:
        if x < 0:
            return -np.inf
        k, l = O
        ek, el = np.exp(O)
        lx = np.log(x)
        return k - ((x / el) ** ek) - ek * l - lx + ek * lx

    @staticmethod
    def ldk(x: float, O: params) -> float:
        if x < 0:
            return -np.inf
        ek, el = np.exp(O)
        xl = x / el
        return 1.0 - ek * ((xl**ek) - 1.0) * np.log(xl)

    @staticmethod
    def ldl(x: float, O: params) -> float:
        if x < 0:
            return -np.inf
        ek, el = np.exp(O)
        return ek * ((x / el) ** ek - 1.0)

    @staticmethod
    def ldO(x: float, O: params) -> np.ndarray:
        return np.array((WeibullModelExp.ldk(x, O), WeibullModelExp.ldl(x, O)))


class GaussianModel(Model):
    """
    f(x) = e^(-1/2 * ((x - m) / sd)^2) / (sd * sqrt(2pi))

    sd = e^(_sd)

    O = {m, _sd}
    """
    @staticmethod
    def name() -> str:
        return "Gaussian"

    @staticmethod
    def params_convert_to_model(O: params) -> params:
        return np.array([O[0], np.log(O[1])])

    @staticmethod
    def params_convert_from_model(O: params) -> params:
        return np.array([O[0], np.exp(O[1])])

    @staticmethod
    def generate(O: params, size: int = 1) -> sample:
        return np.array(norm.rvs(loc=O[0], scale=O[1], size=size))

    @staticmethod
    def p(x: float, O: params) -> float:
        m, sd = O
        sd = np.exp(sd)
        return np.exp(-0.5 * (((x - m) / sd) ** 2)) / (sd * np.sqrt(2 * np.pi))

    @staticmethod
    def lp(x: float, O: params) -> float:
        p = GaussianModel.p(x, O)
        if p <= 0:
            return -np.inf
        return np.log(p)

    @staticmethod
    def ldm(x: float, O: params) -> float:
        m, sd = O
        return (x - m) / (np.exp(2 * sd))

    @staticmethod
    def ldsd(x: float, O: params) -> float:
        m, sd = O
        return ((x - m) ** 2) / np.exp(2 * sd) - 1

    @staticmethod
    def ldO(x: float, O: params) -> np.ndarray:
        return np.array((GaussianModel.ldm(x, O), GaussianModel.ldsd(x, O)))
