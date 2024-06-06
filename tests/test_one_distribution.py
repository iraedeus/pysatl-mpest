"""Unit test module which tests mixture of one distribution parameter estimation"""

import pytest
import numpy as np

from em_algo.models import (
    WeibullModelExp,
    GaussianModel,
    ExponentialModel,
    AModelWithGenerator,
)
from em_algo.distribution import Distribution
from em_algo.mixture_distribution import MixtureDistribution
from em_algo.problem import Problem
from tests.utils import run_test


@pytest.mark.parametrize(
    "model, params, start_params, size, deviation, expected_error",
    [
        (WeibullModelExp(), (0.5, 0.5), (1.0, 1.0), 500, 0.01, 0.05),
        (WeibullModelExp(), (0.3, 1.0), (0.1, 2.0), 500, 0.01, 0.05),
        (GaussianModel(), (0.0, 5.0), (1.0, 5.0), 500, 0.01, 0.1),
        (GaussianModel(), (1.0, 5.0), (0.0, 1.0), 500, 0.01, 0.1),
        (ExponentialModel(), (1.0,), (0.5,), 500, 0.01, 0.05),
        (ExponentialModel(), (2.0,), (3.0,), 500, 0.01, 0.05),
    ],
)
def test_one_distribution(
    model: AModelWithGenerator,
    params,
    start_params,
    size: int,
    deviation: float,
    expected_error: float,
):
    """Runs mixture of one distribution parameter estimation unit test"""

    # pylint: disable=too-many-arguments

    np.random.seed(42)

    params = np.array(params)
    start_params = np.array(start_params)

    x = model.generate(params, size)

    c_params = model.params_convert_to_model(params)
    c_start_params = model.params_convert_to_model(start_params)

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, c_start_params)]
        ),
    )

    for result in run_test(problem=problem, deviation=deviation):
        assert result.error is None

        result_params = result.content.distributions[0].params
        assert float(np.sum(np.abs(c_params - result_params))) <= expected_error
