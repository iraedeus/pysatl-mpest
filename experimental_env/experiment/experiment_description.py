""" A module providing a class that describes a second stage experiment."""

from experimental_env.experiment.result_description import ResultDescription
from mpest import MixtureDistribution, Samples


class ExperimentDescription:
    """
    A class containing the results of parameter estimation, the base mixture, as well as samples
    """

    def __init__(
        self,
        base_mixture: MixtureDistribution,
        result_descr: ResultDescription,
        samples: Samples,
        method_name: str,
    ):
        self._base_mixture = base_mixture
        self._steps = result_descr
        self._samples = samples
        self._method_name = method_name

    @property
    def base_mixture(self):
        """
        Base mixture property
        """

        return self._base_mixture

    @property
    def steps(self):
        """
        A property containing the results of each step of the algorithm.
        """

        return self._steps

    @property
    def samples(self):
        """
        Samples property
        """

        return self._samples

    @property
    def method_name(self):
        """
        Method name property
        """

        return self._method_name

    def get_mixture_name(self):
        """
        Get name of mixture.
        """
        "".join(d.model.name for d in self._base_mixture)
