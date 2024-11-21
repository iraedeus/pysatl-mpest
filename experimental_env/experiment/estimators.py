"""Estimators for estimating parameters in second stage of experiment"""

import multiprocessing
from abc import abstractmethod
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor

from tqdm import tqdm

from experimental_env.utils import choose_best_mle
from mpest import Problem
from mpest.em import EM
from mpest.em.methods.l_moments_method import IndicatorEStep, LMomentsMStep
from mpest.em.methods.likelihood_method import BayesEStep, LikelihoodMStep
from mpest.em.methods.method import Method
from mpest.optimizers import ALL_OPTIMIZERS
from mpest.utils import ANamed, Factory, ResultWithLog

METHODS = {
    "Likelihood": [
        [Factory(BayesEStep), Factory(LikelihoodMStep, optimizer)]
        for optimizer in ALL_OPTIMIZERS
    ],
    "L-moments": [Factory(IndicatorEStep), Factory(LMomentsMStep)],
}

CPU_COUNT = multiprocessing.cpu_count()


class AEstimator(ANamed):
    """
    An abstract class to describe the estimator.
    Implements the second stage of the experiment, evaluating the parameters of the mixture.
    """

    @abstractmethod
    def estimate(self, problems: list[Problem]) -> list[ResultWithLog]:
        """
        The process of estimating the parameters of the mixture
        """
        raise NotImplementedError


class LikelihoodEstimator(AEstimator):
    """
    An estimator using the maximum likelihood method with various optimizers.
    During the estimating process, the result with the best approximation is returned.
    """

    def __init__(
        self, brkpointer: EM.ABreakpointer, dst_checker: EM.ADistributionChecker
    ):
        """
        Class constructor
        """
        self._brkpointer = brkpointer
        self._dst_checker = dst_checker

    @property
    def name(self):
        return "Likelihood"

    def helper(self, problem):
        """
        Helper function for multiprocessed estimation
        """
        methods = [
            Method(step[0].construct(), step[1].construct())
            for step in METHODS["Likelihood"]
        ]
        ems = [EM(self._brkpointer, self._dst_checker, method) for method in methods]
        results = [em.solve_logged(problem, True, True, True) for em in ems]
        return choose_best_mle(problem.distributions, results)

    def estimate(self, problems: list[Problem]) -> list[ResultWithLog]:
        output = []
        print("Starting Likelihood estimation")
        with tqdm(total=len(problems)) as pbar:
            with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                fs = [executor.submit(self.helper, problem) for problem in problems]

                for res in as_completed(fs):
                    pbar.update()
                    output.append(res.result())
        return output


class LMomentsEstimator(AEstimator):
    """
    An estimator using the L-moments method.
    """

    def __init__(self, brkpointer, dst_checker, seed: int = 42):
        self._brkpointer = brkpointer
        self._dst_checker = dst_checker
        self._seed = seed

    @property
    def name(self):
        return "LMoments"

    def helper(self, problem):
        """
        Helper function for multiprocessed estimation
        """
        steps = METHODS["L-moments"]
        new_method = Method(steps[0].construct(), steps[1].construct())
        em_factory = Factory(EM, self._brkpointer, self._dst_checker, new_method)

        return em_factory.construct().solve_logged(problem, True, True, True)

    def estimate(self, problems: list[Problem]) -> list[ResultWithLog]:
        output = []
        print("Starting L-moments estimation")
        with tqdm(total=len(problems)) as pbar:
            with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                fs = [
                    executor.submit(self.helper, problem)
                    for i, problem in enumerate(problems)
                ]
                for f in as_completed(fs):
                    output.append(f.result())
                    pbar.update(1)

        return output
