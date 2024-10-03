"""Class for saving the dataset"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experimental_env.preparation.dataset_description import DatasetDescrciption
from mpest.mixture_distribution import MixtureDistribution
from mpest.types import Samples


class DatasetSaver:
    """
    A class that saves the dataset along the path selected by the user.
    """

    def __init__(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True)

        self._out_dir = path

    def save_dataset(self, descr: DatasetDescrciption) -> None:
        """
        Save dataset
        """
        samples, mixture = descr.samples, descr.base_mixture
        mixture_name_dir: Path = self._out_dir.joinpath(descr.get_dataset_name())
        if not mixture_name_dir.exists():
            mixture_name_dir.mkdir()

        # Checking the experiment number
        experiment_dir: Path = Path()
        for i in range(1000):
            experiment_dir = mixture_name_dir.joinpath(f"experiment_{i}")
            if experiment_dir.exists():
                continue
            experiment_dir.mkdir()
            break

        # Save samples
        samples_file: Path = experiment_dir.joinpath("samples.csv")
        np.savetxt(samples_file, samples, delimiter=",")

        # Save config
        config_file: Path = experiment_dir.joinpath("config.yaml")
        with open(config_file, "w", encoding="utf-8") as config:
            yaml.dump(descr.to_yaml_format(), config, default_flow_style=False)

        # Save config
        hist_file = experiment_dir.joinpath("hist.png")
        plt.hist(samples, density=True, color="lightsteelblue")
        plt.savefig(hist_file)
        plt.close()
