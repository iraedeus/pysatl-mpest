"""
Module which represents distribution mixture.

Distribution mixture can be defined as
p(x | list[pdf], list[params], list[prior_probability]) = sum(prior_probability * pdf(x | params))
"""

from typing import Sized, Iterable, Iterator

from mpest.distribution import Distribution
from mpest.models import AModel
from mpest.types import Params
from mpest.utils import IteratorWrapper


class DistributionInMixture(Distribution):
    """
    Class which represents distribution is mixture.

    Distribution in mixture has an additional parameter which is prior probability
    pdf'(x) = prior_probability * pdf(x)

    Prior probability can be None value, which means that
    this distribution is not in mixture. When prior probability is None value
    it's the same as prior probability is equal zero.
    """

    def __init__(
        self,
        model: AModel,
        params: Params,
        prior_probability: float | None,
    ) -> None:
        super().__init__(model, params)
        self._prior_probability = prior_probability

    @property
    def prior_probability(self):
        """Prior probability getter."""
        return self._prior_probability

    def pdf(self, x: float):
        """Probability density function for distribution in mixture."""
        if self.prior_probability is None:
            return 0.0
        return self.prior_probability * super().pdf(x)


class MixtureDistribution(Sized, Iterable[DistributionInMixture]):
    """
    Class which represents distributions mixture.

    Distributions mixture contains independent distributions
    and their prior probabilities.
    Also can be used as container of distributions in mixture.
    """

    def __init__(
        self,
        distributions: list[DistributionInMixture],
    ) -> None:
        self._distributions = distributions
        self._normalize()

    @classmethod
    def from_distributions(
        cls,
        distributions: list[Distribution],
        prior_probabilities: list[float | None] | None = None,
    ) -> "MixtureDistribution":
        """
        Creates DistributionsMixture object from distributions and their prior probabilities.
        """

        if prior_probabilities is None:
            k = len(distributions)
            prior_probabilities = list([1 / k] * k)
        elif len(prior_probabilities) != len(distributions):
            raise ValueError(
                "Sizes of 'distributions' and 'prior_probabilities' must be the same"
            )

        return cls(
            [
                DistributionInMixture(d.model, d.params, p)
                for d, p in zip(distributions, prior_probabilities)
            ]
        )

    def _normalize(self):
        """
        Normalizing method, which is used to maintain the invariant:
        sum of prior probabilities is equal to zero.
        """

        s = [d.prior_probability for d in self._distributions if d.prior_probability]
        if len(s) == 0:
            return
        s = sum(s)

        prior_probabilities = [
            d.prior_probability / s if d.prior_probability else None
            for d in self._distributions
        ]

        self._distributions = [
            DistributionInMixture(d.model, d.params, p)
            for d, p in zip(self._distributions, prior_probabilities)
        ]

    @property
    def distributions(self) -> list[DistributionInMixture]:
        """Distributions in mixture getter."""
        return self._distributions

    def __iter__(self) -> Iterator[DistributionInMixture]:
        def iterate(instance: MixtureDistribution, index: int):
            if index >= len(instance.distributions):
                raise StopIteration
            return instance.distributions[index]

        return IteratorWrapper(self, iterate)

    def __getitem__(self, ind: int):
        return self.distributions[ind]

    def __len__(self):
        return len(self.distributions)

    def pdf(self, x: float) -> float:
        """Probability density function for distributions mixture."""
        return sum(d.pdf(x) for d in self.distributions)
