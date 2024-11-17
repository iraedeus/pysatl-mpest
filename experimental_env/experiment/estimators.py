"""Estimators for estimating parameters in second stage of experiment"""

import multiprocessing
from abc import abstractmethod
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from experimental_env.utils import choose_best_mle
from mpest import Problem
from mpest.em import EM
from mpest.em.methods.l_moments_method import IndicatorEStep, LMomentsMStep
from mpest.em.methods.likelihood_method import BayesEStep, LikelihoodMStep
from mpest.em.methods.method import Method
from mpest.optimizers import ALL_OPTIMIZERS
from mpest.utils import ANamed, ResultWithLog

METHODS = {
    "Likelihood": [
        Method(BayesEStep(), LikelihoodMStep(optimizer)) for optimizer in ALL_OPTIMIZERS
    ],
    "L-moments": Method(IndicatorEStep(), LMomentsMStep()),
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
        self.ems = [
            EM(brkpointer, dst_checker, method) for method in METHODS["Likelihood"]
        ]

    @property
    def name(self):
        return "Likelihood"

    def estimate(self, problems: list[Problem]) -> list[ResultWithLog]:
        output = np.zeros((len(problems), 0))
        print("Starting Likelihood estimation")
        with tqdm(total=len(problems) * len(ALL_OPTIMIZERS)) as pbar:
            with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                for em in self.ems:
                    fs = [
                        executor.submit(em.solve_logged, problem, True, True, True)
                        for problem in problems
                    ]
                    res = []
                    for f in as_completed(fs):
                        res.append(f.result())
                        pbar.update(1)

                    output = np.column_stack((output, np.array(res)))

        return [choose_best_mle(results) for results in output.tolist()]


class LMomentsEstimator(AEstimator):
    """
    An estimator using the L-moments method.
    """

    def __init__(self, brkpointer, dst_checker):
        self.em = EM(brkpointer, dst_checker, METHODS["L-moments"])

    @property
    def name(self):
        return "LMoments"

    def estimate(self, problems: list[Problem]) -> list[ResultWithLog]:
        output = []
        print("Starting L-moments estimation")
        with tqdm(total=len(problems)) as pbar:
            for problem in problems:
                res = self.em.solve_logged(problem, True, True, True)
                output.append(res)
                pbar.update(1)

        return output
