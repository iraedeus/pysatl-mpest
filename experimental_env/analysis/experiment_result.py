from experimental_env.experiment.result_description import ExperimentDescription
from mpest import MixtureDistribution, Samples


class ExtendedResultDescription:
    def __init__(
        self,
        base_mixture: MixtureDistribution,
        result_descr: ExperimentDescription,
        samples: Samples,
        method_name: str,
    ):
        self._base_mixture = base_mixture
        self._steps = result_descr
        self._samples = samples
        self._method_name = method_name

    @property
    def base_mixture(self):
        return self._base_mixture

    @property
    def steps(self):
        return self._steps

    @property
    def samples(self):
        return self._samples

    @property
    def method_name(self):
        return self._method_name

    def get_mixture_name(self):
        "".join(d.model.name for d in self._base_mixture)
