""" A module containing classes for storing information about the results of estimating at the second stage """

from typing import Iterable, Iterator

from mpest import MixtureDistribution
from mpest.utils import ResultWithLog


class StepDescription:
    """
    A class containing information about each step of the algorithm
    """

    def __init__(self, mixture, time):
        self._result_mixture = mixture
        self._time = time

    @property
    def result_mixture(self) -> MixtureDistribution:
        """
        The property containing the mixture from this step.
        """
        return self._result_mixture

    @property
    def time(self) -> float:
        """
        A property containing the time that was spent to complete this step of the algorithm
        """
        return self._time

    def to_yaml_format(self) -> dict:
        """
        A function for converting information to yaml format
        """
        output = {}

        # Get params of distributions
        dists = {}
        for d in self._result_mixture:
            dists[d.model.name] = {
                "params": d.params.tolist(),
                "prior": float(d.prior_probability),
            }
        output["distributions"] = dists
        output["time"] = self._time

        return output


class ResultDescription(Iterable):
    """
    A class containing information about all the steps of the algorithm.
    """

    def __init__(self):
        self._steps = []

    @classmethod
    def from_result(cls, result: ResultWithLog):
        """
        Class method for initializing a class through an estimating result with logs
        """
        instance = cls()
        instance._steps = [
            StepDescription(step.result.content, step.time) for step in result.log.log
        ]
        return instance

    @classmethod
    def from_steps(cls, steps: list[StepDescription]):
        """
        Class method for initializing a class through the estimating results of each step
        """
        instance = cls()
        instance._steps = steps
        return instance

    @property
    def steps(self) -> list[StepDescription]:
        """
        A property that contains all the steps of the EM algorithm in this experiment
        """
        return self._steps

    def __next__(self):
        return self._steps[0]

    def __iter__(self) -> Iterator:
        """
        Iterating over steps
        """
        return iter(self._steps)
