"""A module representing error classes"""


class EStepError(Exception):
    """
    A class representing an error that occurs in E step
    """


class MStepError(Exception):
    """
    A class representing an error that occurs in M step
    """


class SampleError(EStepError):
    """
    A class representing an error that occurs when the entire sample does not match our mixture
    """
