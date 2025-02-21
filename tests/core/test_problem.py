import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from mpest import MixtureDistribution, Problem
from tests.core.utils import generate_random_distribution


@settings(max_examples=1000)
@given(st.integers(2, 10))
def test_problem_init(size):
    samples = np.array([1.0, 2.0, 3.0])
    priors = np.random.dirichlet(np.ones(size))
    dists = [
        generate_random_distribution(params_range=(0.0, 5.0), prior_probability_range=(prior, prior), in_mixture=True)
        for prior in priors
    ]
    mixture = MixtureDistribution(dists)

    problem = Problem(samples, mixture)

    assert np.array_equal(problem.samples, samples)
    assert all(np.array_equal(d.params, dmx.params) for d, dmx in zip(dists, mixture.distributions))
    assert all(np.isclose(d.prior_probability, dmx.prior_probability) for d, dmx in zip(dists, mixture.distributions))
    assert isinstance(problem.distributions, MixtureDistribution)
