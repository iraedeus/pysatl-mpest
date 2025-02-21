import random

import numpy as np

from mpest import Distribution, DistributionInMixture
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp


def generate_random_distribution(
    model_types=None,
    params_range=(0.0, 10.0),
    prior_probability_range=(0.1, 1.0),
    in_mixture=False,
):
    if model_types is None:
        model_types = [GaussianModel, ExponentialModel, WeibullModelExp]

    model_type = random.choice(model_types)
    model = model_type()

    num_params = {
        GaussianModel: 2,
        ExponentialModel: 1,
        WeibullModelExp: 2,
    }.get(model_type, 1)

    params = np.random.uniform(params_range[0], params_range[1], size=num_params)

    if in_mixture:
        if prior_probability_range:
            prior_probability = np.random.uniform(prior_probability_range[0], prior_probability_range[1])
        else:
            prior_probability = None

        return DistributionInMixture(model, params, prior_probability)
    else:
        return Distribution(model, params)
