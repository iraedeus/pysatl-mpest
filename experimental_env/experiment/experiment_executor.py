"""A module with classes performing the second stage of the experiment."""

import random
import warnings
from collections import OrderedDict
from pathlib import Path

from experimental_env.experiment.estimators import AEstimator
from experimental_env.experiment.experiment_saver import ExperimentSaver
from experimental_env.experiment.result_description import ResultDescription
from experimental_env.preparation.dataset_saver import DatasetSaver
from experimental_env.utils import create_random_mixture
from mpest import Problem
from mpest.models import ALL_MODELS


class RandomExperimentExecutor:
    """
    A performer who randomly generates the initial conditions for the algorithm.
    """

    def __init__(self, path: Path, seed: int = 42):
        """
        Class constructor

        :param path: The path in which the results of the second stage of the experiment will lie
        :param seed: Seed for determined results.
        """
        self._out_dir = path
        self._seed = 42

        random.seed(seed)

    def execute(self, preparation_results: dict, estimator: AEstimator) -> None:
        """
        Function for the execution of the second stage

        :param preparation_results: Data from the first stage received from the parser.
        :param estimator: Estimator
        """
        method_dir: Path = self._out_dir.joinpath(estimator.name)

        for mixture_name in preparation_results.keys():
            mixture_name_dir: Path = method_dir.joinpath(mixture_name)
            descriptions = OrderedDict(preparation_results[mixture_name].items())
            models = [
                ALL_MODELS[d.model.name]
                for d in next(iter(descriptions.values())).base_mixture
            ]

            problems = [
                Problem(item[1].samples, create_random_mixture(models, self._seed + i))
                for i, item in enumerate(descriptions.items())
            ]

            # Disable warnings and estimating params.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = estimator.estimate(problems)

            # Saving results
            for i, item in enumerate(descriptions.items()):
                exp_dir: Path = mixture_name_dir.joinpath(item[0])
                DatasetSaver(exp_dir).save_dataset(item[1])

                descr = ResultDescription.from_result(results[i])
                ExperimentSaver(exp_dir).save(descr)
