"""Module from generating datasets with the given sample size, experimental counts and mixture"""
import random
from pathlib import Path

from tqdm import tqdm

from experimental_env.preparation.dataset_saver import DatasetDescrciption, DatasetSaver
from experimental_env.utils import create_random_mixture
from mpest.distribution import Distribution
from mpest.mixture_distribution import MixtureDistribution
from mpest.models.abstract_model import AModel


class RandomDatasetGenerator:
    """
    Class that generates datasets from mixture.
    Randomize params of base mixture at each experiment.
    You can select the sample size and the number of experiments.
    """

    def __init__(self, seed: int = 42):
        """
        Setting seed for determined result.
        """
        random.seed(seed)

    def generate(
        self,
        samples_size: int,
        models: list[type[AModel]],
        working_path: Path,
        exp_count: int = 100,
    ):
        """
        A function that generates datasets based on random mixture.
        """

        with tqdm(total=exp_count) as tbar:
            for _ in range(exp_count):
                tbar.update()
                mixture = create_random_mixture(models)
                samples = mixture.generate(samples_size)

                descr = DatasetDescrciption(samples_size, samples, mixture)
                saver = DatasetSaver(working_path)
                saver.save_dataset(descr)


class ConcreteDatasetGenerator:
    """
    A preparation class that allows you to generate datasets based on user-selected mixtures
    """

    _dists = []
    _priors = []

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def add_distribution(
        self, model: type[AModel], params: list[float], prior: float
    ) -> None:
        """
        Add distribution with params and prior to mixture.
        """
        self._dists.append(Distribution.from_params(model, params))
        self._priors.append(prior)

    def generate(self, samples_size: int, working_path: Path):
        """
        A function that generates a dataset based on a user's mixture.
        """

        mixture = MixtureDistribution.from_distributions(self._dists, self._priors)
        samples = mixture.generate(samples_size)

        descr = DatasetDescrciption(samples_size, samples, mixture)
        saver = DatasetSaver(working_path)
        saver.save_dataset(descr)
