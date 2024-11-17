from scipy.stats import yulesimon

from mpest.utils import ResultWithLog


class StepDescription:
    def __init__(self, step):
        self.result_mixture = step.result.result
        self.time = step.time

    def to_yaml_format(self):
        pass


class ExperimentDescription:
    def __init__(self, result: ResultWithLog):
        self._steps = [StepDescription(step) for step in result.log.log]

    @property
    def steps(self):
        return self._steps

    def __iter__(self):
        for step in self._steps:
            yield step