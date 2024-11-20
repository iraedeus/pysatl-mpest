"""Functions for experimental environment"""
from abc import ABC, abstractmethod
from random import uniform

import numpy as np

from mpest import Distribution, MixtureDistribution
from mpest.models import AModel, ExponentialModel, GaussianModel
from mpest.utils import ResultWithLog


class AMixtureGenerator(ABC):
    @abstractmethod
    def generate_priors(self, models):
        """
        a
        """

    def create_random_mixture(
        self, models: list[type[AModel]], seed: int = 42
    ) -> MixtureDistribution:
        """
        Function for generating random mixture
        """

        dists = []
        np.random.seed(seed)
        priors = self.generate_priors(models)
        for m in models:
            if m == ExponentialModel:
                params = [uniform(0.1, 5)]
            elif m == GaussianModel:
                params = [uniform(-5, 5), uniform(0.1, 5)]
            else:
                params = [uniform(0.1, 5), uniform(0.1, 5)]

            dists.append(Distribution.from_params(m, params))

        return MixtureDistribution.from_distributions(dists, priors)


class DatasetMixtureGenerator(AMixtureGenerator):
    def generate_priors(self, models):
        return np.random.dirichlet(alpha=[1.7 for _ in range(len(models))], size=1)[0]


class ExperimentMixtureGenerator(AMixtureGenerator):
    def generate_priors(self, models):
        return [1 / len(models) for _ in range(len(models))]


def choose_best_mle(results: list[ResultWithLog]) -> ResultWithLog:
    """
    The method for choosing the best result in the maximum likelihood method
    """

    return results[0]
