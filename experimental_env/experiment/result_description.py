""" A module containing classes for storing information about the results of the second stage """

from typing import Iterable, Iterator

from mpest import MixtureDistribution
from mpest.utils import ResultWithLog


class StepDescription:
    """
    A class containing information about each step of the algorithm
    """

    def __init__(self, step):
        self._result_mixture = step.result.content
        self._time = step.time

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


class ExperimentDescription(Iterable):
    """
    A class containing information about all the steps of the algorithm.
    """

    def __init__(self, result: ResultWithLog):
        self._steps = [StepDescription(step) for step in result.log.log]

    @property
    def steps(self) -> list[StepDescription]:
        """
        A property that contains all the steps of the EM algorithm in this experiment
        """
        return self._steps

    def __iter__(self) -> Iterator:
        """
        Iterating over steps
        """
        return iter(self._steps)
