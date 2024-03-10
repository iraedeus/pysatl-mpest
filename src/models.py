from abc import ABC, abstractmethod
import numpy as np
from utils import *


class Model(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
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
    @staticmethod
    def name() -> str:
        return "WeibullExp"

    @staticmethod
    def p(x: float, O: params) -> float:
        if x < 0:
            return 0
        ek, el = np.exp(O)
        xl = x / el
        return (ek / el) * (xl ** (ek - 1.0)) * (np.exp(-(xl**ek)))

    @staticmethod
    def lp(x: float, O: params) -> float:
        k, l = O
        ek, el = np.exp(O)
        lx = np.log(x)
        return k - ((x / el) ** ek) - ek * l - lx + ek * lx

    @staticmethod
    def ldk(x: float, O: params) -> float:
        ek, el = np.exp(O)
        xl = x / el
        return 1.0 - ek * ((xl**ek) - 1.0) * np.log(xl)

    @staticmethod
    def ldl(x: float, O: params) -> float:
        ek, el = np.exp(O)
        return ek * ((x / el) ** ek - 1.0)

    @staticmethod
    def ldO(x: float, O: params) -> np.ndarray:
        return np.array((WeibullModelExp.ldk(x, O), WeibullModelExp.ldl(x, O)))


class GaussianModel(Model):
    @staticmethod
    def name() -> str:
        return "Gaussian"

    @staticmethod
    def modify(O: params) -> params:
        return np.array([O[0], np.exp(O[1])])

    @staticmethod
    def p(x: float, O: params) -> float:
        m, sd = GaussianModel.modify(O)
        return np.exp(-0.5 * (((x - m) / sd) ** 2)) / (sd * np.sqrt(2 * np.pi))

    @staticmethod
    def lp(x: float, O: params) -> float:
        p = GaussianModel.p(x, O)
        if p == 0:
            return 0
        return np.log(p)

    @staticmethod
    def ldm(x: float, O: params) -> float:
        m, sd = GaussianModel.modify(O)
        return (x - m) / (sd**2)

    @staticmethod
    def ldst(x: float, O: params) -> float:
        m, sd = GaussianModel.modify(O)
        sd2 = sd**2
        return ((m**2) - (2 * m * x) - sd2 + (x**2)) / (sd2 * sd)

    @staticmethod
    def ldO(x: float, O: params) -> np.ndarray:
        return np.array((GaussianModel.ldm(x, O), GaussianModel.ldst(x, O)))
