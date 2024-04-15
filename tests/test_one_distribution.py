"""TODO"""

import pytest
import numpy as np

from em_algo.models import WeibullModelExp, GaussianModel, ExponentialModel, Model
from em_algo.em import EM
from em_algo.distribution import Distribution


@pytest.mark.parametrize(
    "model, params, start_params, size, deviation, expected_error",
    [
        (WeibullModelExp, (0.5, 0.5), (1.0, 1.0), 500, 0.01, 0.05),
        (WeibullModelExp, (0.3, 1.0), (0.1, 2.0), 500, 0.01, 0.05),
        (GaussianModel, (0.0, 5.0), (1.0, 5.0), 500, 0.01, 0.1),
        (GaussianModel, (1.0, 5.0), (0.0, 1.0), 500, 0.01, 0.1),
        (ExponentialModel, (1.0,), (0.5,), 500, 0.01, 0.05),
        (ExponentialModel, (2.0,), (3.0,), 500, 0.01, 0.05),
    ],
    ids=[
        "Weibull (0.5, 0.5)",
        "Weibull (0.3, 1.0)",
        "Gaussian (0.0, 5.0)",
        "Gaussian (1.0, 5.0)",
        "Exponential (1.0,)",
        "Exponential (2.0,)",
    ],
)
def test_one_distribution(
    model: type[Model],
    params,
    start_params,
    size: int,
    deviation: float,
    expected_error: float,
):
    """TODO"""

    np.random.seed(42)

    params = np.array(params)
    start_params = np.array(start_params)

    x = model.generate(params, size)

    c_params = model.params_convert_to_model(params)
    c_start_params = model.params_convert_to_model(start_params)

    result = EM.em_algo(
        samples=x,
        distributions=[Distribution(model, c_start_params)],
        k=1,
        deviation=deviation,
    )

    result_params = result.distributions[0].params
    assert float(np.sum(np.abs(c_params - result_params))) <= expected_error
