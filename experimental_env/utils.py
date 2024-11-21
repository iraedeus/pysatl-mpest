"""Functions for experimental environment"""
from abc import ABC, abstractmethod
from random import uniform

import numpy as np

from experimental_env.analysis.metrics import Parametric
from mpest import Distribution, MixtureDistribution
from mpest.models import ALL_MODELS, AModel, ExponentialModel, GaussianModel
from mpest.utils import ResultWithLog


class AMixtureGenerator(ABC):
    """
    An abstract class for generating mixtures.
    """

    @abstractmethod
    def generate_priors(self, models):
        """
        A method for choosing how a prior probabilities will be generated.
        The function is needed so that, with a uniform distribution, degenerate distributions are not generated.
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
                params = [uniform(0, 5), uniform(0.1, 5)]
            else:
                params = [uniform(0.1, 5), uniform(0.1, 5)]

            dists.append(Distribution.from_params(m, params))

        return MixtureDistribution.from_distributions(dists, priors)


class DatasetMixtureGenerator(AMixtureGenerator):
    """
    A class for generating a prior probabilities for datasets.
    The alphas for the Dirichlet distribution are 5, which means that a prior probabilities will rarely be close to 0 or 1
    """

    def generate_priors(self, models):
        return np.random.dirichlet(alpha=[5 for _ in range(len(models))], size=1)[0]


class ExperimentMixtureGenerator(AMixtureGenerator):
    """
    A class for generating a prior probabilities for init mixtures in experiment stage.
    """

    def generate_priors(self, models):
        return [1 / len(models) for _ in range(len(models))]


def create_mixture_by_key(config: dict, key: str) -> MixtureDistribution:
    """
    A function that simplifies the creation of a mix from a config file.
    """
    # Read even one element as list
    if not isinstance(config[key], list):
        config[key] = [config[key]]

    priors = [d["prior"] for d in config[key]]
    dists = [
        Distribution.from_params(ALL_MODELS[d["type"]], d["params"])
        for d in config[key]
    ]

    return MixtureDistribution.from_distributions(dists, priors)


def choose_best_mle(
    base_mixture: MixtureDistribution, results: list[ResultWithLog]
) -> ResultWithLog:
    """
    The method for choosing the best result in the maximum likelihood method
    """
    best_result = results[0]

    for result in results:
        res_mixture = result.content.content
        if Parametric().error(base_mixture, res_mixture) < Parametric().error(
            base_mixture, best_result.content.content
        ):
            best_result = result

    return best_result
