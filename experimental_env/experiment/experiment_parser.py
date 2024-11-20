""" A module containing a class for parsing the second stage of the experiment"""

import os
from pathlib import Path

import yaml
from numpy import genfromtxt

from experimental_env.analysis.analysis import ParserOutput
from experimental_env.experiment.experiment_description import ExperimentDescription
from experimental_env.experiment.result_description import (
    ResultDescription,
    StepDescription,
)
from mpest import Distribution, MixtureDistribution
from mpest.models import ALL_MODELS


class ExperimentParser:
    """
    A class that parses the second stage of the experiment.
    """

    def get_base_mixture(self, config_p: Path) -> MixtureDistribution:
        """
        Get base mixture from config
        """
        with open(config_p, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            # Read even one component mixture as list
            if not isinstance(config["distributions"], list):
                config["distributions"] = [config["distributions"]]

            priors = [d["prior"] for d in config["distributions"]]
            dists = [
                Distribution.from_params(ALL_MODELS[d["type"]], d["params"])
                for d in config["distributions"]
            ]

            return MixtureDistribution.from_distributions(dists, priors)

    def parse(self, path: Path) -> ParserOutput:
        """
        The parsing method

        :param path: The path to the directory of the second stage of the experiment
        """
        output = {}

        # Open each mixture dir
        for mixture_name in os.listdir(path):
            mixture_name_dir: Path = path.joinpath(mixture_name)
            mixture_name_list = []

            # Open each experiment dir
            for exp_name in os.listdir(mixture_name_dir):
                experiment_dir: Path = mixture_name_dir.joinpath(exp_name)
                samples_p: Path = experiment_dir.joinpath("samples.csv")
                config_p: Path = experiment_dir.joinpath("config.yaml")

                # Get samples
                samples = genfromtxt(samples_p, delimiter=",")

                # Get base mixture
                base_mixture = self.get_base_mixture(config_p)

                # Get all steps
                steps = []
                for step in os.listdir(experiment_dir):
                    if not "step" in step:
                        continue
                    step_dir = experiment_dir.joinpath(step)
                    step_config_p = step_dir.joinpath("config.yaml")

                    # Get step
                    with open(step_config_p, "r", encoding="utf-8") as config_file:
                        config = yaml.safe_load(config_file)
                        error = config["error"]
                        step_time = config["time"]

                        priors = [d["prior"] for d in config["distributions"]]
                        dists = [
                            Distribution.from_params(
                                ALL_MODELS[d["type"]], d["params"]
                            )
                            for d in config["distributions"]
                        ]

                        step_mixture = MixtureDistribution.from_distributions(
                            dists, priors
                        )

                    steps.append(StepDescription(step_mixture, step_time))

                # Save results of experiment
                mixture_name_list.append(
                    ExperimentDescription(
                        base_mixture,
                        ResultDescription.from_steps(steps, error),
                        samples,
                        "OFF",
                    )
                )

            output[mixture_name] = mixture_name_list
        return output
