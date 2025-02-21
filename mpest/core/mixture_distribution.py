"""Module representing a mixture of probability distributions.

This module defines the `MixtureDistribution` class, which encapsulates a collection of independent
distributions with their prior probabilities, and the `DistributionInMixture` class, which extends
a single distribution with a prior probability. The mixture's probability density function (PDF)
is the weighted sum of individual PDFs, and sample generation is supported if all active
distributions have generators.
"""

from collections.abc import Iterable, Iterator, Sized

import numpy as np

from mpest.annotations import Params
from mpest.core.distribution import APDFAble, AWithGenerator, Distribution
from mpest.models import AModel, AModelWithGenerator
from mpest.utils import IteratorWrapper


class DistributionInMixture(Distribution):
    """Represents a single distribution within a mixture, with an associated prior probability.

    Extends the `Distribution` class by adding a prior probability, which scales the PDF and
    determines the distribution's contribution to the mixture. A `None` prior probability indicates
    the distribution is inactive (equivalent to a prior of 0.0).

    Attributes:
        _prior_probability (float | None): The prior probability of this distribution in the mixture.
    """

    def __init__(
        self,
        model: AModel,
        params: Params,
        prior_probability: float | None,
    ) -> None:
        """Initializes a DistributionInMixture instance.

        Args:
            model (AModel): The statistical model of the distribution (e.g., GaussianModel).
            params (Params): A NumPy array of parameters in the model's internal format.
            prior_probability (float | None): The prior probability of this distribution in the mixture.
                If `None`, the distribution is considered inactive (treated as 0.0).
        """

        super().__init__(model, params)
        self._prior_probability = prior_probability

    @property
    def prior_probability(self):
        """Gets the prior probability of the distribution in the mixture.

        Returns:
            float | None: The prior probability, or `None` if the distribution is inactive.
        """

        return self._prior_probability

    def pdf(self, x: float):
        """Evaluates the weighted probability density function at a given point.
        The PDF is calculated as the prior probability multiplied by the base model's PDF:

        .. math::
            p_j(x \\mid \theta_j, \\omega_j) = \\omega_j f_j(x \\mid \theta_j)

        where:
        - :math:`\\omega_j` is the prior probability,
        - :math:`f_j(x \\mid \theta_j)` is the base model's PDF,
        - :math:`\theta_j` are the parameters of the distribution.

        If the prior probability is `None`, returns 0.0.

        Args:
            x (float): The point at which to evaluate the PDF.

        Returns:
            float: The weighted PDF value at `x`.

        Examples:
            >>> from mpest.models import GaussianModel
            >>> dist = DistributionInMixture(GaussianModel(), np.array([0.0, 0.0]), None)
            >>> dist.pdf(1.0)  # Inactive distribution returns 0.0
            0.0
        """

        if self.prior_probability is None:
            return 0.0
        return self.prior_probability * super().pdf(x)


class MixtureDistribution(APDFAble, AWithGenerator, Sized, Iterable[DistributionInMixture]):
    """Represents a mixture of independent probability distributions.

    A mixture distribution combines multiple `DistributionInMixture` objects, where the total PDF is
    the sum of each distribution's PDF weighted by its prior probability. The class normalizes prior
    probabilities to sum to 1 (excluding inactive distributions) and supports sample generation if
    all active distributions have generators.

    Attributes:
        _distributions (list[DistributionInMixture]): The list of distributions in the mixture.
    """

    # TODO: Think about empty mixture: raise TypeError or adapt methods to it

    def __init__(
        self,
        distributions: list[DistributionInMixture],
    ) -> None:
        """Initializes a MixtureDistribution instance.

        Args:
            distributions (list[DistributionInMixture]): A list of `DistributionInMixture` objects
                defining the mixture.
        """

        self._distributions = distributions
        self._normalize()

    @classmethod
    def from_distributions(
        cls,
        distributions: list[Distribution],
        prior_probabilities: list[float | None] | None = None,
    ) -> "MixtureDistribution":
        """Creates a MixtureDistribution from a list of distributions and prior probabilities.

        Provides a convenient way to initialize a mixture from raw `Distribution` objects and their
        prior probabilities. If prior probabilities are not provided, they are set to equal weights
        (1/k, where k is the number of distributions).

        Args:
            distributions (list[Distribution]): A list of `Distribution` objects to include in the mixture.
            prior_probabilities (list[float | None] | None): A list of prior probabilities for each
                distribution. If `None`, defaults to equal probabilities (1/k). If provided, must match
                the length of `distributions`. Normalized if their sum exceeds 1 (excluding `None` values).

        Returns:
            MixtureDistribution: A new mixture distribution instance.

        Raises:
            ValueError: If the lengths of `distributions` and `prior_probabilities` do not match when
                `prior_probabilities` is provided.
        """

        if prior_probabilities is None:
            k = len(distributions)
            prior_probabilities = list([1 / k] * k)
        elif len(prior_probabilities) != len(distributions):
            raise ValueError("Sizes of 'distributions' and 'prior_probabilities' must be the same")

        return cls([DistributionInMixture(d.model, d.params, p) for d, p in zip(distributions, prior_probabilities)])

    def __iter__(self) -> Iterator[DistributionInMixture]:
        """Returns an iterator over the distributions in the mixture.

        Returns:
            Iterator[DistributionInMixture]: An iterator yielding each `DistributionInMixture` object.
        """

        def iterate(instance: MixtureDistribution, index: int):
            if index >= len(instance.distributions):
                raise StopIteration
            return instance.distributions[index]

        return IteratorWrapper(self, iterate)

    def __getitem__(self, ind: int):
        """Gets the distribution at the specified index.

        Args:
            ind (int): The index of the distribution to retrieve.

        Returns:
            DistributionInMixture: The distribution at the specified index.

        Raises:
            IndexError: If `ind` is out of range.
        """

        return self.distributions[ind]

    def __len__(self):
        """Gets the number of distributions in the mixture.

        Returns:
            int: The total number of distributions (active and inactive).
        """

        return len(self.distributions)

    def pdf(self, x: float) -> float:
        r"""Evaluates the probability density function of the mixture at a given point.

        The PDF is computed as the sum of each distribution's weighted PDF:

        .. math::
            p(x \mid \Omega, F, \Theta) = \sum_{j=1}^k \omega_j f_j(x \mid \theta_j)

        where:
        - :math:`p(x \mid \Omega, F, \Theta)` is the mixture PDF,
        - :math:`f_j(x \mid \theta_j)` is the PDF of the j-th distribution,
        - :math:`\omega_j` is the prior probability of the j-th distribution,
        - :math:`\theta_j` are the parameters of the j-th distribution,
        - :math:`k` is the number of distributions.

        Args:
            x (float): The point at which to evaluate the PDF.

        Returns:
            float: The mixture PDF value at `x`.
        """

        return sum(d.pdf(x) for d in self.distributions)

    @property
    def has_generator(self):
        """Checks if the mixture supports sample generation.

        Returns True if all active distributions (those with non-`None` prior probabilities) have
        generators (i.e., their models are instances of `AModelWithGenerator`).

        Returns:
            bool: True if sample generation is supported, False otherwise.

        Examples:
            >>> from mpest.models import GaussianModel
            >>> class Dummy(AModel):
            >>>     pass

            >>> d1 = DistributionInMixture(GaussianModel(), np.array([0.0, 0.0]), 0.5)
            >>> d2 = DistributionInMixture(Dummy(), np.array([1.0, 0.0]), None)
            >>> mixture = MixtureDistribution([d1, d2])
            >>> mixture.has_generator
            True
        """

        return all(
            isinstance(d.model, AModelWithGenerator) for d in self.distributions if d.prior_probability is not None
        )

    def generate(self, size=1) -> Params:
        """Generates random samples from the mixture distribution.

        Samples are generated by probabilistically selecting distributions based on their prior
        probabilities and then sampling from the chosen distributions. Only active distributions
        (with non-`None` prior probabilities) are considered.

        Args:
            size (int): The number of samples to generate. Defaults to 1.

        Returns:
            Params: A NumPy array of generated samples.

        Raises:
            TypeError: If any active distribution lacks a generator.
        """

        models: list[tuple[Distribution, float]] = []
        for d in self.distributions:
            if d.prior_probability is not None:
                if d.has_generator:
                    models.append((d, d.prior_probability))
                else:
                    raise TypeError(f"Model {d.model} hasn't got a generator")

        x_models = list(
            np.random.choice(
                list(range(len(models))),
                p=[model[1] for model in models],
                size=size,
            )
        )

        temp = []
        for i, model in enumerate(models):
            temp += list(model[0].generate(x_models.count(i)))

        x = np.array(temp)
        np.random.shuffle(x)

        return x

    def _normalize(self):
        """Normalizes the prior probabilities of active distributions to sum to 1.

        Adjusts the prior probabilities of distributions with non-`None` values so their sum equals 1.
        Inactive distributions (with `None` prior probabilities) remain unchanged.

        Examples:
            >>> from mpest.models import GaussianModel
            >>> d1 = DistributionInMixture(GaussianModel(), np.array([0.0, 0.0]), 2.0)
            >>> d2 = DistributionInMixture(GaussianModel(), np.array([1.0, 0.0]), 3.0)
            >>> mixture = MixtureDistribution([d1, d2])
            >>> mixture[0].prior_probability
            0.4
            >>> mixture[1].prior_probability
            0.6
        """

        s = [d.prior_probability for d in self._distributions if d.prior_probability]
        if len(s) == 0:
            return
        s = sum(s)

        prior_probabilities = [d.prior_probability / s if d.prior_probability else None for d in self._distributions]

        self._distributions = [
            DistributionInMixture(d.model, d.params, p) for d, p in zip(self._distributions, prior_probabilities)
        ]

    @property
    def distributions(self) -> list[DistributionInMixture]:
        """Gets the list of distributions in the mixture.

        Returns:
            list[DistributionInMixture]: The list of `DistributionInMixture` objects, including both
                active and inactive distributions.
        """

        return self._distributions
