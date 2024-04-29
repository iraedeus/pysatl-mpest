"""TODO"""

from typing import Sized, Iterable, Iterator, TypeVar

from em_algo.distribution import Distribution
from em_algo.models import AModel
from em_algo.types import Params
from em_algo.utils import IteratorWrapper

T = TypeVar("T", bound=Distribution)


class DistributionInMixture(Distribution):
    """TODO"""

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
        """TODO"""
        return self._prior_probability


class DistributionMixture(Sized, Iterable[DistributionInMixture]):
    """TODO"""

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
    ) -> "DistributionMixture":
        """TODO"""

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
        """TODO"""

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
        """TODO"""
        return self._distributions

    def __iter__(self) -> Iterator[DistributionInMixture]:

        def iterate(instance: DistributionMixture, index: int):
            if index >= len(instance.distributions):
                raise StopIteration
            return instance.distributions[index]

        return IteratorWrapper(self, iterate)

    def __getitem__(self, ind: int):
        return self.distributions[ind]

    def __len__(self):
        return len(self.distributions)

    def pdf(self, x: float) -> float:
        """TODO"""
        s = 0
        for d in self.distributions:
            if d.prior_probability is not None:
                s += d.prior_probability * d.model.p(x, d.params)
        return s
