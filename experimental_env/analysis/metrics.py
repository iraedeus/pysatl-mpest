from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np

from mpest import MixtureDistribution
from mpest.utils import ANamed


class AMetric(ANamed, ABC):
    @abstractmethod
    def single_error(
        self, base_mixture: MixtureDistribution, result_mixture: MixtureDistribution
    ) -> float:
        pass

    @abstractmethod
    def error(
        self,
        base_mixtures: list[MixtureDistribution],
        result_mixtures: list[MixtureDistribution],
    ) -> list[float]:
        pass


class MSE(AMetric):
    @property
    def name(self) -> str:
        return "MSE"

    def single_error(self, base_mixture, result_mixture):
        pass

    def error(self, base_mixtures, result_mixtures):
        return sum(
            [
                self.single_error(base, result)
                for base, result in zip(base_mixtures, result_mixtures)
            ]
        ) / len(result_mixtures)


class Parametric(AMetric):
    @property
    def name(self) -> str:
        return "ParamsError"

    def single_error(self, base_mixture, result_mixture):
        base_p, res_p = (
            [d.params for d in ld] for ld in (base_mixture, result_mixture)
        )

        param_diff = min(
            sum(sum(abs(x - y)) for x, y in zip(base_p, _res_p))
            for _res_p in permutations(res_p)
        )

        return param_diff

    def error(self, base_mixtures, result_mixtures):
        return sum(
            [
                self.single_error(base, result)
                for base, result in zip(base_mixtures, result_mixtures)
            ]
        ) / len(result_mixtures)


class PDFMetric(AMetric):
    def __init__(self, a, b):
        self.start = a
        self.end = b

    @property
    def name(self) -> str:
        return "PDFError"

    def single_error(
        self, base_mixture: MixtureDistribution, result_mixture: MixtureDistribution
    ) -> float:
        X = np.linspace(self.start, self.end, 30)
        errors = [abs(base_mixture.pdf(x) - result_mixture.pdf(x)) for x in X]

        return sum(errors)

    def error(self, base_mixtures, result_mixtures):
        return sum(
            [
                self.single_error(base, result)
                for base, result in zip(base_mixtures, result_mixtures)
            ]
        ) / len(result_mixtures)


METRICS = {
    MSE().name: MSE,
    Parametric().name: Parametric,
    PDFMetric(-np.inf, np.inf).name: PDFMetric,
}
