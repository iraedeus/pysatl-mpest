"""Unit test module which tests mixture of two distributions parameter estimation"""

# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

import numpy as np
import pytest

from mpest.distribution import Distribution
from mpest.mixture_distribution import MixtureDistribution
from mpest.models import (
    AModelWithGenerator,
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)
from mpest.problem import Problem
from mpest.utils import Factory
from tests.utils import (
    check_for_params_error_tolerance,
    run_test,
    check_for_priors_error_tolerance,
)


@pytest.mark.parametrize(
    "model_factory, params, start_params, prior_probability, size, deviation,"
    "expected_params_error, expected_priors_error",
    [
        (
            Factory(WeibullModelExp),
            [(0.5, 1.0), (1.0, 0.5)],
            [(1.0, 0.5), (0.5, 1.0)],
            [0.3, 0.7],
            500,
            0.01,
            0.2,
            0.1,
        ),
        (
            Factory(WeibullModelExp),
            [(0.5, 0.5), (1.0, 1.0)],
            [(0.5, 1.0), (1.5, 1.5)],
            [0.66, 0.33],
            500,
            0.01,
            0.5,
            0.1,
        ),
        (
            Factory(GaussianModel),
            [(0.0, 5.0), (1.0, 1.0)],
            [(1.0, 3.0), (0.0, 3.0)],
            [0.1, 0.9],
            500,
            0.01,
            0.35,
            0.1,
        ),
        (
            Factory(GaussianModel),
            [(4.0, 5.0), (3.0, 2.0)],
            [(6.0, 6.0), (5.0, 3.5)],
            [0.3, 0.7],
            500,
            0.01,
            0.4,
            0.2,
        ),
        (
            Factory(ExponentialModel),
            [(1.0,), (2.0,)],
            [(0.2,), (1.0,)],
            [0.3, 0.7],
            500,
            0.01,
            0.2,
            0.2,
        ),
        (
            Factory(ExponentialModel),
            [(2.0,), (5.0,)],
            [(1.0,), (7.0,)],
            [0.645, 0.355],
            500,
            0.01,
            0.2,
            0.3,
        ),
    ],
)
def test_two_same_distributions_simple(
    model_factory: Factory[AModelWithGenerator],
    params,
    start_params,
    prior_probability: list[float],
    size: int,
    deviation: float,
    expected_params_error,
    expected_priors_error,
):
    """Runs mixture of two distributions parameter estimation unit test"""

    np.random.seed(42)

    models = [model_factory.construct() for _ in range(len(params))]

    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    c_params = [
        model.params_convert_to_model(param) for model, param in zip(models, params)
    ]
    c_start_params = [
        model.params_convert_to_model(param)
        for model, param in zip(models, start_params)
    ]

    base_mixture = MixtureDistribution.from_distributions(
        [Distribution(model, param) for model, param in zip(models, c_params)],
        prior_probability,
    )

    x = base_mixture.generate(size)

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, param) for model, param in zip(models, c_start_params)]
        ),
    )

    results = run_test(problem=problem, deviation=deviation)
    assert check_for_params_error_tolerance(
        results, base_mixture, expected_params_error
    )
    assert check_for_priors_error_tolerance(
        results, base_mixture, expected_priors_error
    )
