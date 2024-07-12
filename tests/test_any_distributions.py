"""Unit test module which tests mixture of several different distribution parameter estimation"""

import numpy as np
import pytest

from mpest import MixtureDistribution, Distribution, Problem
from mpest.models import WeibullModelExp, ExponentialModel, GaussianModel
from tests.utils import run_test, check_for_error_tolerance


@pytest.mark.parametrize(
    "models, params, start_params, prior_probabilities, size, deviation, expected_error",
    [
        (
            [WeibullModelExp, GaussianModel],
            [[0.5, 1.0], [5.0, 1.0]],
            [[1.5, 1.0], [1.0, 2.0]],
            [0.33, 0.66],
            1500,
            0.01,
            0.1,
        ),
        (
            [ExponentialModel, GaussianModel],
            [[0.5], [5.0, 1.0]],
            [[1.0], [0.0, 5.0]],
            [0.33, 0.66],
            1500,
            0.01,
            0.1,
        ),
        (
            [ExponentialModel, GaussianModel, WeibullModelExp],
            [[1.0], [5.0, 1.0], [4.0, 1.0]],
            [[1.0], [1.0, 5.0], [1.0, 2.0]],
            [0.35, 0.3, 0.35],
            1500,
            0.01,
            0.3,
        ),
    ],
)
def test(
    models,
    params,
    start_params,
    prior_probabilities,
    size,
    deviation,
    expected_error,
):
    """Runs mixture of several different distributions parameter estimation unit test"""

    # pylint: disable=too-many-arguments

    np.random.seed(42)

    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    c_params = [
        model().params_convert_to_model(param) for model, param in zip(models, params)
    ]
    c_start_params = [
        model().params_convert_to_model(param)
        for model, param in zip(models, start_params)
    ]

    base_mixture = MixtureDistribution.from_distributions(
        [Distribution(model(), param) for model, param in zip(models, c_params)],
        prior_probabilities,
    )

    x = base_mixture.generate(size)

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [
                Distribution(model(), param)
                for model, param in zip(models, c_start_params)
            ]
        ),
    )

    results = run_test(problem=problem, deviation=deviation)
    assert check_for_error_tolerance(results, base_mixture, expected_error)
