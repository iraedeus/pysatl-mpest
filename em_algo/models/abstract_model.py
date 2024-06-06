"""Module which contains abstract model classes, which describe distribution models"""

from abc import ABC, abstractmethod
import numpy as np

from em_algo.types import Samples, Params
from em_algo.utils import Named


class AModel(Named, ABC):
    """Abstract class which represents all methods needed from model for EM algorithm"""

    @abstractmethod
    def params_convert_to_model(self, params: Params) -> Params:
        """
        Method which converts generally accepted distribution params
        into local ones, for handling optimizers which needs continuous distributions
        """

    @abstractmethod
    def params_convert_from_model(self, params: Params) -> Params:
        """
        Method which converts local distribution params into
        generally accepted ones, for handling optimizers which needs continuous distributions
        """

    @abstractmethod
    def pdf(self, x: float, params: Params) -> float:
        """Probability density function"""

    @abstractmethod
    def lpdf(self, x: float, params: Params) -> float:
        """
        Logarithm of probability density function,
        for speed improvements if it possible
        """


class AModelDifferentiable(AModel, ABC):
    """Abstract class which extends AModel by adding derivatives"""

    @abstractmethod
    def ld_params(self, x: float, params: Params) -> np.ndarray:
        """
        Method which returns list of logarithms of partial derivatives
        with respects to all it's params
        """


class AModelWithGenerator(AModel, ABC):
    """
    Abstract class which extends AModel by adding ability to generate samples
    by given params
    """

    @abstractmethod
    def generate(self, params: Params, size: int = 1) -> Samples:
        """Method which generates samples by given params"""
