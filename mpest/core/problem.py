"""Module representing a problem for parameter estimation of a mixture distribution.

This module defines the `Problem` class, which encapsulates the data (samples) and the initial mixture
distribution approximation to be solved using the EM algorithm. It also defines the abstract `ASolver`
class, which provides an interface for solvers (e.g., EM algorithm implementations) to estimate the
parameters of the mixture distribution.
"""

from abc import ABC, abstractmethod

from mpest.annotations import Samples
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.utils import ResultWithError


class Problem:
    """Represents a parameter estimation problem for a mixture distribution.

    This class combines observational data (samples) with an initial approximation of a mixture
    distribution, serving as the input to solvers like the EM algorithm. The goal is to refine
    the parameters of the mixture distribution to better fit the samples.

    Attributes:
        _samples (Samples): The observational data as a NumPy array.
        _distributions (MixtureDistribution): The initial mixture distribution approximation.
    """

    def __init__(
        self,
        samples: Samples,
        distributions: MixtureDistribution,
    ) -> None:
        """Initializes a Problem instance.

        Args:
            samples (Samples): A NumPy array of observational data points used for parameter estimation.
            distributions (MixtureDistribution): The initial mixture distribution approximation.
        """
        # TODO: Check types of samples and mixture. Maybe rename 'distributions' to 'mixture'

        self._samples = samples
        self._distributions = distributions

    @property
    def samples(self):
        """Gets the observational data samples.

        Returns:
            Samples: A NumPy array containing the observational data points.
        """

        return self._samples

    @property
    def distributions(self):
        """Gets the initial mixture distribution approximation.

        Returns:
            MixtureDistribution: The mixture distribution object containing the initial approximation.
        """

        return self._distributions


Result = ResultWithError[MixtureDistribution]
"""Type alias for the result of solving a `Problem`.

Represents the outcome of a solver, containing a `MixtureDistribution` (the final state of the mixture)
and an optional error. If the solver completes successfully, the `error` field is `None`. If an error
occurs, the `error` field contains the exception, and the `MixtureDistribution` reflects the state
at the point of failure. See `ResultWithError` for details on accessing the result and error."""


class ASolver(ABC):
    """Abstract base class for solvers of mixture distribution parameter estimation problems.

    This class defines the interface that all solver implementations (e.g., EM algorithm variants)
    must follow to solve a `Problem` instance. Subclasses must implement the `solve` method to
    estimate the parameters of the mixture distribution based on the provided samples.
    """

    @abstractmethod
    def solve(self, problem: Problem) -> Result:
        """Solves the parameter estimation problem for a mixture distribution.

        Args:
            problem (Problem): The problem instance containing samples and an initial mixture distribution.

        Returns:
            Result: A `ResultWithError` object containing the final `MixtureDistribution` and an optional
                error. If no error occurs, `error` is `None`. If an error occurs, `error` contains the
                exception, and the `MixtureDistribution` represents the state at the failure point.
        """
