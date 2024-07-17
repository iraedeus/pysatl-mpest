"""Module which represents E and M step classes with different methods."""

from functools import partial

import numpy as np

from mpest.distribution import Distribution
from mpest.mixture_distribution import MixtureDistribution
from mpest.models import AModel, AModelDifferentiable
from mpest.optimizers import AOptimizerJacobian, ScipyTNC, TOptimizer
from mpest.problem import Problem
from mpest.types import Samples
from mpest.utils import ResultWithError


class ExpectationStep:
    """
    Class which represents E step for EM method.

    :param method: Method which use for E step
    :type method: str
    """

    def __init__(self, method: str):
        self.method = method

    def likelihood_method_step(self, problem: Problem) -> tuple:
        """
        A function that represents the E step for the maximum likelihood method
        Calculates the matrix for the M step.

        :param problem: Object of class Problem, which contains samples and mixture.
        :type problem: Problem

        :return: Tuple with args for M step
        """

        samples = problem.samples
        mixture = problem.distributions
        p_xij = []
        active_samples = []
        for x in samples:
            p = np.array([d.model.pdf(x, d.params) for d in mixture])
            if np.any(p):
                p_xij.append(p)
                active_samples.append(x)

        if not active_samples:
            return ResultWithError(mixture, Exception("All models can't match"))

        # h[j, i] contains probability of X_i to be a part of distribution j
        m = len(p_xij)
        k = len(mixture)
        h = np.zeros([k, m], dtype=float)
        curr_w = np.array([d.prior_probability for d in mixture])
        for i, p in enumerate(p_xij):
            wp = curr_w * p
            swp = np.sum(wp)

            if not swp:
                return ResultWithError(mixture, Exception("Error in E step"))
            h[:, i] = wp / swp

        return (active_samples, h, problem)

    def l_moments_step(self, problem: Problem) -> tuple:
        """
        A function that represents the E step for the L moments method.
        Initialize indicators for components of mixture, and recalculate prior probabilities.
        """

    def step(self, problem: Problem):
        """
        A function for conveniently calling the desired method
        """

        if self.method == "likelihood":
            return self.likelihood_method_step(problem)
        if self.method == "l_moments":
            return self.l_moments_step(problem)
        raise ValueError("Unknown method")


class MaximizationStep:
    """
    Class which represents M step for EM method.

    :param method: Method which use for E step
    :type method: str
    """

    def __init__(self, method: str):
        self.method = method

    def likelihood_method_step(
        self,
        samples: Samples,
        h: np.array,
        problem: Problem,
        optimizer: TOptimizer,
    ) -> MixtureDistribution:
        """
        A function that uses the logarithm of likelihood
        to recalculate parameters and prior probabilities

        :param samples: Active samples of mixture
        :param h: The matrix with which the parameters of the mixture will be recalculated
        :param problem: problem which contains all samples and mixture
        :param optimizer: An optimizer that will maximize the logarithm of likelihood

        :return: Mixture with new params
        """

        m = len(h[0])
        mixture = problem.distributions
        new_w = np.sum(h, axis=1) / m
        new_distributions: list[Distribution] = []
        for j, ch in enumerate(h[:]):
            d = mixture[j]

            def log_likelihood(params, ch, model: AModel):
                return -np.sum(ch * [model.lpdf(x, params) for x in samples])

            def jacobian(params, ch, model: AModelDifferentiable):
                return -np.sum(
                    ch
                    * np.swapaxes([model.ld_params(x, params) for x in samples], 0, 1),
                    axis=1,
                )

            # maximizing log of likelihood function for every active distribution
            if isinstance(optimizer, AOptimizerJacobian):
                if not isinstance(d.model, AModelDifferentiable):
                    raise TypeError

                new_params = optimizer.minimize(
                    partial(log_likelihood, ch=ch, model=d.model),
                    d.params,
                    jacobian=partial(jacobian, ch=ch, model=d.model),
                )
            else:
                new_params = optimizer.minimize(
                    func=partial(log_likelihood, ch=ch, model=d.model),
                    params=d.params,
                )

            new_distributions.append(Distribution(d.model, new_params))

        return MixtureDistribution.from_distributions(new_distributions, new_w)

    def l_moments_step(
        self,
    ) -> MixtureDistribution:
        """
        A function that represents M step with L moments method.
        Function recalculate parameters using new prior probabilities from E step.

        :return: Mixture with new params
        """

    def step(self, args, optimizer: TOptimizer = ScipyTNC()) -> MixtureDistribution:
        """
        A function for conveniently calling the desired method
        """

        if self.method == "likelihood":
            return self.likelihood_method_step(*args, optimizer=optimizer)
        if self.method == "l_moments":
            return self.l_moments_step(*args, optimizer=optimizer)
        raise ValueError("Unknown method")
