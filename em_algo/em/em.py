"""
Module which represents EM algorithm and few of it's params:
- EM.Breakpointer
- EM.DistributionChecker
"""

from functools import partial
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from em_algo.types import Samples
from em_algo.distribution import Distribution
from em_algo.distribution_mixture import DistributionMixture, DistributionInMixture
from em_algo.utils import (
    logged,
    ObjectWrapper,
    TimerResultWrapper,
    ResultWithError,
    ResultWithLog,
)
from em_algo.problem import Problem, Result, ASolver
from em_algo.optimizers import TOptimizer, AOptimizerJacobian
from em_algo.models import AModel, AModelDifferentiable
from em_algo.utils import Named


class EM(ASolver):
    """Class which represents EM algorithm"""

    class ABreakpointer(Named, ABC):
        """Abstract class which represents EM breakpointer function handler"""

        @abstractmethod
        def is_over(
            self,
            step: int,
            previous_step: DistributionMixture | None,
            current_step: DistributionMixture,
        ) -> bool:
            """Breakpointer function"""

    class ADistributionChecker(Named, ABC):
        """
        Abstract class which represents distribution checker function handler.
        Used for dynamically removing of degenerate distribution in mixture.
        """

        @abstractmethod
        def is_alive(
            self,
            step: int,
            distribution: DistributionInMixture,
        ) -> bool:
            """
            Distribution checker function,
            which returns True if distribution is correct and not degenerated
            """

    class _DistributionMixtureAlive(DistributionMixture):
        """
        Class which represents distribution mixture which in EM algorithm solving process.
        Remembers the distributions order and control over the degenerated ones.
        """

        def __init__(
            self,
            distributions: list[DistributionInMixture],
            distribution_alive: Callable[[DistributionInMixture], bool] | None = None,
        ) -> None:
            super().__init__(distributions)

            self._checker: Callable[[DistributionInMixture], bool] = (
                distribution_alive if distribution_alive else lambda _: True
            )

            self._active_indexes = list(range(len(self._distributions)))
            self._active = self._distributions

            if distribution_alive:
                self.update(self)

        @classmethod
        def from_distributions(
            cls,
            distributions: list[Distribution],
            prior_probabilities: list[float | None] | None = None,
            distribution_alive: Callable[[DistributionInMixture], bool] | None = None,
        ) -> "EM._DistributionMixtureAlive":
            return cls(
                list(
                    DistributionMixture.from_distributions(
                        distributions,
                        prior_probabilities,
                    )
                ),
                distribution_alive,
            )

        @property
        def distributions(self) -> list[DistributionInMixture]:
            """Active (non degenerated) distributions getter"""
            return self._active

        @property
        def all_distributions(self) -> DistributionMixture:
            """All distributions getter"""
            return DistributionMixture(self._distributions)

        def update(
            self,
            distribution_mixture: DistributionMixture,
            distribution_checker: Callable[[DistributionInMixture], bool] | None = None,
        ):
            """
            Updating active distributions by given one.
            Applies distribution checker function to active distributions.
            Removes degenerated ones from active and sets their prior probabilities to None.
            """

            if distribution_checker is not None:
                self._checker = distribution_checker

            if len(distribution_mixture) != len(self._active):
                raise ValueError(
                    "New distribution mixture size must be the same with previous"
                )

            new_active_indexes: list[int] = []
            for ind, d in zip(self._active_indexes, distribution_mixture):
                if self._checker(d):
                    new_active_indexes.append(ind)
                    self._distributions[ind] = d
                else:
                    self._distributions[ind] = DistributionInMixture(
                        d.model,
                        d.params,
                        None,
                    )

            self._normalize()
            self._active_indexes = new_active_indexes
            self._active = [self._distributions[ind] for ind in new_active_indexes]

    def __init__(
        self,
        breakpointer: "EM.ABreakpointer",
        distribution_checker: "EM.ADistributionChecker",
        optimizer: TOptimizer,
    ):
        self.breakpointer = breakpointer
        self.distribution_checker = distribution_checker
        self.optimizer = optimizer

    class Log:
        """Class which represents EM algorithm log object"""

        class Item:
            """Class which represents EM algorithm log object for intermediate step"""

            def __init__(
                self,
                result: ResultWithError[DistributionMixture] | None,
                time: float | None,
            ) -> None:
                self._result = result
                self._time = time

            @property
            def result(self):
                """Logged step result getter"""
                return self._result

            @property
            def time(self):
                """Logged runtime getter"""
                return self._time

        def __init__(
            self,
            log: list[
                TimerResultWrapper[ResultWithError[DistributionMixture]]
                | ObjectWrapper[ResultWithError[DistributionMixture]]
                | float
            ],
            steps: int,
        ) -> None:
            self._log: list[EM.Log.Item] = []
            for note in log:
                if isinstance(note, float | int):
                    self._log.append(EM.Log.Item(None, note))
                elif isinstance(note, TimerResultWrapper):
                    self._log.append(EM.Log.Item(note.content, note.runtime))
                else:
                    self._log.append(EM.Log.Item(note.content, None))
            self._steps = steps

        @property
        def log(self):
            """Log getter"""
            return self._log

        @property
        def steps(self):
            """Count of steps getter"""
            return self._steps

    @staticmethod
    def step(
        samples: Samples,
        distribution_mixture: DistributionMixture,
        optimizer: TOptimizer,
    ) -> ResultWithError[DistributionMixture]:
        """EM algo step"""

        # E part

        p_xij = []
        active_samples = []
        for x in samples:
            p = np.array([d.model.pdf(x, d.params) for d in distribution_mixture])
            if np.any(p):
                p_xij.append(p)
                active_samples.append(x)

        if not active_samples:
            return ResultWithError(
                distribution_mixture, Exception("All models can't match")
            )

        # h[j, i] contains probability of X_i to be a part of distribution j
        m = len(p_xij)
        k = len(distribution_mixture)
        h = np.zeros([k, m], dtype=float)
        curr_w = np.array([d.prior_probability for d in distribution_mixture])
        for i, p in enumerate(p_xij):
            wp = curr_w * p
            swp = np.sum(wp)

            if not swp:
                return ResultWithError(
                    distribution_mixture, Exception("Error in E step")
                )
            h[:, i] = wp / swp

        # M part

        # Need attention due creating all w==np.nan problem
        # instead of removing distribution which is a cause of error

        new_w = np.sum(h, axis=1) / m
        new_distributions: list[Distribution] = []
        for j, ch in enumerate(h[:]):
            d = distribution_mixture[j]

            def log_likelihood(params, ch, model: AModel):
                return -np.sum(ch * [model.lpdf(x, params) for x in active_samples])

            def jacobian(params, ch, model: AModelDifferentiable):
                return -np.sum(
                    ch
                    * np.swapaxes(
                        [model.ld_params(x, params) for x in active_samples], 0, 1
                    ),
                    axis=1,
                )

            # maximizing log of likelihood function for every active distribution
            if isinstance(optimizer, AOptimizerJacobian):
                if not isinstance(d.model, AModelDifferentiable):
                    return ResultWithError(
                        distribution_mixture,
                        ValueError(
                            f"Model {d.model.name} can't handle optimizer with jacobian."
                        ),
                    )
                new_params = optimizer.minimize(
                    partial(log_likelihood, ch=ch, model=d.model),
                    d.params,
                    jacobian=partial(jacobian, ch=ch, model=d.model),
                )
            else:
                new_params = optimizer.minimize(
                    partial(log_likelihood, ch=ch, model=d.model),
                    d.params,
                )

            new_distributions.append(Distribution(d.model, new_params))

        return ResultWithError(
            DistributionMixture.from_distributions(
                new_distributions,
                new_w,
            )
        )

    def solve_logged(
        self,
        problem: Problem,
        create_history: bool = False,
        remember_time: bool = False,
    ) -> ResultWithLog[ResultWithError[DistributionMixture], Log]:
        """
        The parameter estimation of distribution mixture problem solver,
        which uses EM algorithm and allows to collect logs of each algorithm step result.
        """

        history = []

        def log_map(distributions: ResultWithError[EM._DistributionMixtureAlive]):
            return ResultWithError(
                distributions.content.all_distributions,
                distributions.error,
            )

        @logged(
            history,
            save_results=create_history,
            save_results_mapper=log_map,
            save_time=remember_time,
        )
        def make_step(
            step: int,
            distributions: EM._DistributionMixtureAlive,
        ) -> ResultWithError[EM._DistributionMixtureAlive]:
            """EM algorithm full step with checking distributions"""

            result = EM.step(problem.samples, distributions, self.optimizer)
            if result.error:
                return ResultWithError(
                    distributions,
                    result.error,
                )

            distributions.update(
                result.content,
                lambda d: self.distribution_checker.is_alive(step, d),
            )

            error = (
                Exception("All distributions failed")
                if len(distributions) == 0
                else None
            )

            return ResultWithError(distributions, error)

        previous_step = None
        distributions = EM._DistributionMixtureAlive(list(problem.distributions))
        step = 0

        while not self.breakpointer.is_over(step, previous_step, distributions):
            previous_step = distributions.all_distributions
            if make_step(step, distributions).content.error:
                break
            step += 1

        return ResultWithLog(
            ResultWithError(distributions.all_distributions),
            EM.Log(history, step),
        )

    def solve(self, problem: Problem) -> Result:
        return self.solve_logged(problem, False, False).content
