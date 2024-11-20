""" A module containing classes for storing information about the results of estimating at the second stage """
import math
from types import NoneType
from typing import Iterable, Iterator

import numpy as np

from experimental_env.preparation.dataset_description import DatasetDescrciption
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
        dists = []
        for d in self._result_mixture:
            prior = float(d.prior_probability) if d.prior_probability else np.nan
            dists.append(
                {
                    "type": d.model.name,
                    "params": d.params.tolist(),
                    "prior": prior,
                }
            )
        output["step_distributions"] = dists
        output["time"] = self._time

        return output


class ResultDescription(Iterable):
    """
    A class containing information about all the steps of the algorithm.
    """

    def __init__(
        self,
        init_mixture: MixtureDistribution,
        result: ResultWithLog,
        ds_descr: DatasetDescrciption,
    ):
        self._init_mixture = init_mixture
        self._steps = [
            StepDescription(step.result.content, step.time) for step in result.log.log
        ]
        self._ds_descr = ds_descr
        self._error = True if result.log.log[-1].result.error else False

    @property
    def steps(self) -> list[StepDescription]:
        """
        A property that contains all the steps of the EM algorithm in this experiment
        """
        return self._steps

    @property
    def error(self):
        return self._error

    def __next__(self):
        return self._steps[0]

    def __iter__(self) -> Iterator:
        """
        Iterating over steps
        """
        return iter(self._steps)

    def to_yaml_format(self):
        output = self._ds_descr.to_yaml_format()

        dists = []
        for d in self._init_mixture:
            dists.append(
                {
                    "type": d.model.name,
                    "params": d.params.tolist(),
                    "prior": float(d.prior_probability),
                }
            )
        output["init_distributions"] = dists

        output["error"] = self._error

        return output
