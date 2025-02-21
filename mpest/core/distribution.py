"""Module representing a single probability distribution.

This module defines the `Distribution` class and related abstract base classes to encapsulate a
probability distribution with its model and parameters. It provides methods to compute the
probability density function (PDF) and generate samples (if supported by the model). The class
is designed to work seamlessly with mixture distributions and the EM algorithm in the `mpest`
library.
"""

from abc import ABC, abstractmethod

import numpy as np

from mpest.annotations import Params, Samples
from mpest.models import AModel, AModelWithGenerator


class APDFAble(ABC):
    """Abstract base class defining the interface for probability density function evaluation.

    Subclasses must implement the `pdf` method to provide the probability density function,
    which is a core requirement for any distribution object in the `mpest` library.
    """

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Evaluates the probability density function at a given point.

        Args:
            x (float): The value at which to evaluate the PDF.

        Returns:
            float: The probability density value at `x`.
        """


class AWithGenerator(ABC):
    """Abstract base class defining the interface for sample generation.

    Subclasses must implement the `generate` method to enable random sample generation from the
    distribution, which is optional but required for models supporting sampling in the `mpest`
    library.
    """

    @abstractmethod
    def generate(self, size=1) -> Samples:
        """Generates random samples from the distribution.

        Args:
            size (int): The number of samples to generate. Defaults to 1.

        Returns:
            Samples: A NumPy array containing the generated samples.
        """


class Distribution(APDFAble, AWithGenerator):
    """Class representing a probability distribution with a model and parameters.

    This class encapsulates a distribution defined by a model (e.g., Gaussian) and its
    parameters. It supports evaluating the PDF and generating samples if the underlying model
    implements a generator.

    Attributes:
        _model (AModel | AModelWithGenerator): The underlying distribution model.
        _params (Params): The parameters of the distribution in the model's internal format.
    """

    def __init__(
        self,
        model: AModel | AModelWithGenerator,
        params: Params,
    ) -> None:
        """Initializes a Distribution object.

        Args:
            model (AModel | AModelWithGenerator): The statistical model defining the distribution
                (e.g., GaussianModel, ExponentialModel).
            params (Params): A NumPy array of parameters in the model's internal format (e.g.,
                [mean, log(sigma)] for Gaussian).

        Examples:
            >>> from mpest.models import GaussianModel
            >>> dist = Distribution(GaussianModel(), np.array([0.0, np.log(1.0)]))
            >>> dist.pdf(1.0)  # Compute PDF at x=1.0
            0.24197072451914337
            >>> dist.generate(3)  # Generate 3 samples
            array([...])  # Random values from N(0, 1)

        TODO:
            Add validation to ensure `params` are compatible with `model`.
        """

        self._model = model
        self._params = params

    @classmethod
    def from_params(
        cls,
        model: type[AModel],
        params: list[float],
    ) -> "Distribution":
        """Creates a Distribution instance from a model type and parameter list.

        This is a user-friendly constructor that initializes a model instance and converts the
        parameter list to a NumPy array.

        Args:
            model (type[AModel]): The class of the model to instantiate (e.g., GaussianModel).
            params (list[float]): A list of parameters in the model's standard format (e.g.,
                [mean, sigma] for Gaussian).

        Returns:
            Distribution: A new Distribution instance.

        Examples:
            >>> from mpest.models import GaussianModel
            >>> dist = Distribution.from_params(GaussianModel, [0.0, 1.0])
            >>> dist.params
            array([0.0, 0.0])  # [mean, log(sigma)]
        """

        model_obj = model()
        return cls(model_obj, np.array(params))

    def pdf(self, x: float):
        """Evaluates the probability density function at a given point.

        The PDF is computed using the underlying model's `pdf` method, with parameters converted
        to the model's internal format.

        Args:
            x (float): The point at which to evaluate the PDF.

        Returns:
            float: The probability density value at `x`.
        """

        return self.model.pdf(x, self.model.params_convert_to_model(self.params))

    @property
    def model(self):
        """Gets the underlying model of the distribution.

        Returns:
            AModel | AModelWithGenerator: The statistical model instance.
        """

        return self._model

    @property
    def params(self):
        """Gets the parameters of the distribution.

        Returns:
            Params: A NumPy array of parameters in the model's internal format.
        """

        return self._params

    @property
    def has_generator(self):
        """Checks if the distribution supports sample generation.

        Returns:
            bool: True if the model is an instance of `AModelWithGenerator`, False otherwise.
        """

        return isinstance(self.model, AModelWithGenerator)

    def generate(self, size=1):
        """Generates random samples from the distribution.

        Uses the underlying model's `generate` method if available, passing parameters in the internal
        format. This method is only functional if the model supports sampling.

        Args:
            size (int): The number of samples to generate. Defaults to 1.

        Returns:
            Samples: A NumPy array containing the generated samples.

        Raises:
            TypeError: If the model does not support sample generation (not an `AModelWithGenerator`).

        Examples:
            If size is 0, output is empty list
            >>> from mpest.models import ExponentialModel
            >>> dist = Distribution(ExponentialModel(), np.array([0.0]))
            >>> empty_samples = dist.generate(0)
            >>> len(empty_samples)
            0
        """

        if not isinstance(self.model, AModelWithGenerator):
            raise TypeError(f"Model {self.model} hasn't got a generator")
        return self.model.generate(self.model.params_convert_to_model(self.params), size=size)
