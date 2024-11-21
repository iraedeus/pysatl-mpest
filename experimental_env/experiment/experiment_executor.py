"""A module with classes performing the second stage of the experiment."""

import random
import warnings
from pathlib import Path

from experimental_env.experiment.estimators import AEstimator
from experimental_env.experiment.experiment_saver import ExperimentSaver
from experimental_env.experiment.result_description import ResultDescription
from experimental_env.utils import ExperimentMixtureGenerator
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

        for mixture_name, ds_descriptions in preparation_results.items():
            mixture_name_dir: Path = method_dir.joinpath(mixture_name)
            models = [ALL_MODELS[model_name] for model_name in mixture_name.split("_")]

            problems = [
                Problem(
                    descr.samples,
                    ExperimentMixtureGenerator().create_random_mixture(
                        models, self._seed + i
                    ),
                )
                for i, descr in enumerate(ds_descriptions)
            ]

            # Disable warnings and estimating params.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = estimator.estimate(problems)

            # Saving results
            for i, ds_descr in enumerate(ds_descriptions):
                exp_dir: Path = mixture_name_dir.joinpath(
                    f"experiment_{ds_descr.exp_num}"
                )
                result = results[i]

                result_descr = ResultDescription.from_result(
                    problems[i].distributions, result, ds_descr
                )
                ExperimentSaver(exp_dir).save(result_descr)
