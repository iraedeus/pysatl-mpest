"""Functions for experimental environment"""

from random import uniform

import numpy as np

from mpest import Distribution, MixtureDistribution
from mpest.models import AModel, ExponentialModel, GaussianModel
from mpest.utils import ResultWithLog


def create_random_mixture(
    models: list[type[AModel]], seed: int = 42
) -> MixtureDistribution:
    """Function for generating random mixture"""

    dists = []
    np.random.seed(seed)
    priors = list(np.random.dirichlet(alpha=np.ones(len(models)), size=1)[0])
    for m in models:
        if m == ExponentialModel:
            params = [uniform(0.1, 5)]
        elif m == GaussianModel:
            params = [uniform(-5, 5), uniform(0.1, 5)]
        else:
            params = [uniform(0.1, 5), uniform(0.1, 5)]

        dists.append(Distribution.from_params(m, params))

    return MixtureDistribution.from_distributions(dists, priors)


def choose_best_mle(results: list[ResultWithLog]) -> ResultWithLog:
    """
    The method for choosing the best result in the maximum likelihood method
    """

    return results[0]
