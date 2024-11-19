import os
from pathlib import Path

import yaml
from numpy import genfromtxt

from experimental_env.analysis.experiment_result import ExtendedResultDescription
from experimental_env.experiment.result_description import (
    ExperimentDescription,
    StepDescription,
)
from mpest import Distribution, MixtureDistribution
from mpest.models import ALL_MODELS


class ExperimentParser:
    def parse(self, path: Path) -> dict:
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

                # Get mixture
                with open(config_p, "r", encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file)
                    samples_size = config["samples_size"]
                    priors = [d["prior"] for d in config["distributions"].values()]
                    dists = [
                        Distribution.from_params(ALL_MODELS[d_name], d_item["params"])
                        for d_name, d_item in config["distributions"].items()
                    ]

                    base_mixture = MixtureDistribution.from_distributions(dists, priors)

                steps = []

                for step in os.listdir(experiment_dir):
                    if not "step" in step:
                        continue
                    step_dir = experiment_dir.joinpath(step)
                    step_config_p = step_dir.joinpath("config.yaml")

                    with open(step_config_p, "r", encoding="utf-8") as config_file:
                        config = yaml.safe_load(config_file)
                        step_time = config["time"]

                        priors = [d["prior"] for d in config["distributions"].values()]
                        dists = [
                            Distribution.from_params(
                                ALL_MODELS[d_name], d_item["params"]
                            )
                            for d_name, d_item in config["distributions"].items()
                        ]

                        step_mixture = MixtureDistribution.from_distributions(
                            dists, priors
                        )

                    steps.append(StepDescription(step_mixture, step_time))

                mixture_name_list.append(
                    ExtendedResultDescription(
                        base_mixture,
                        ExperimentDescription.from_steps(steps),
                        samples,
                        "OFF",
                    )
                )

            output[mixture_name] = mixture_name_list
        return output
