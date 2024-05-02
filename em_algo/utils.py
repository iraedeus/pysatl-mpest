"""TODO"""

import functools
from typing import ParamSpec, TypeVar, Callable, Generic, Iterator
from abc import ABC, abstractmethod

import time


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class IteratorWrapper(Generic[T, R], Iterator[R]):
    """TODO"""

    def __init__(
        self,
        instance: T,
        next_function: Callable[[T, int], R],
    ) -> None:
        self._instance = instance
        self._next_function = next_function
        self._ind = -1

    def __next__(self):
        self._ind += 1
        return self._next_function(self._instance, self._ind)


class Indexed(Generic[T]):
    """TODO"""

    def __init__(
        self,
        ind: int,
        content: T,
    ) -> None:
        self._ind = ind
        self._content = content

    @property
    def ind(self):
        """TODO"""
        return self._ind

    @property
    def content(self):
        """TODO"""
        return self._content


class Named(ABC):
    """TODO"""

    @property
    @abstractmethod
    def name(self) -> str:
        """TODO"""


class Factory(Generic[T]):
    """TODO"""

    def __init__(self, cls: type[T], *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def construct(self):
        """TODO"""
        return self.cls(*self.args, **self.kwargs)


class ResultWrapper(Generic[T]):
    """TODO"""

    def __init__(
        self,
        result: T,
    ) -> None:
        self._result = result

    @property
    def result(self):
        """TODO"""
        return self._result


class ResultWithError(ResultWrapper[T]):
    """TODO"""

    def __init__(
        self,
        result: T,
        error: Exception | None = None,
    ) -> None:
        super().__init__(result)
        self._error = error

    @property
    def error(self):
        """TODO"""
        return self._error


class ResultWithLog(Generic[T, R], ResultWrapper[T]):
    """TODO"""

    def __init__(self, result: T, log: R) -> None:
        super().__init__(result)
        self._log = log

    @property
    def log(self):
        """TODO"""
        return self._log


class TimerResultWrapper(ResultWrapper[T]):
    """TODO"""

    def __init__(self, result: T, runtime: float) -> None:
        super().__init__(result)
        self._runtime = runtime

    @property
    def runtime(self):
        """TODO"""
        return self._runtime


def apply(mapper: Callable[[R], T]):
    """TODO"""

    def current_apply(func: Callable[P, R]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper_apply(*args: P.args, **kwargs: P.kwargs) -> T:
            result = func(*args, **kwargs)
            return mapper(result)

        return wrapper_apply

    return current_apply


def timer(func: Callable[P, R]) -> Callable[P, TimerResultWrapper[R]]:
    """TODO"""

    @functools.wraps(func)
    def wrapper_timer(*args: P.args, **kwargs: P.kwargs) -> TimerResultWrapper[R]:
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        finish = time.perf_counter_ns()
        runtime = (finish - start) * 1e-6
        return TimerResultWrapper(result, runtime)

    return wrapper_timer


def history(holder: list[T], mapper: Callable[[R], T] = lambda x: x):
    """TODO"""

    def current_history(func: Callable[P, R]) -> Callable[P, R]:
        """TODO"""

        @functools.wraps(func)
        def wrapped_history(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            holder.append(mapper(result))
            return result

        return wrapped_history

    return current_history


def logged(
    holder: list[TimerResultWrapper[T] | ResultWrapper[T] | float],
    save_results: bool = True,
    save_results_mapper: Callable[[R], T] = lambda x: x,
    save_time: bool = True,
):
    """TODO"""

    def time_only(x: TimerResultWrapper[R]):
        return x.runtime

    def apply_mapper_to_result(x: ResultWrapper[R]):
        return ResultWrapper(save_results_mapper(x.result))

    def apply_mapper_to_timer_result(x: TimerResultWrapper[R]):
        return TimerResultWrapper(save_results_mapper(x.result), x.runtime)

    def current_logged(func: Callable[P, R]) -> Callable[P, ResultWrapper[R]]:
        """TODO"""

        curr_func = apply(ResultWrapper[R])(func)

        if save_time:
            if not save_results:
                history_decorator = history(holder, mapper=time_only)
            else:
                history_decorator = history(holder, mapper=apply_mapper_to_timer_result)
            curr_func = history_decorator(timer(func))
            return curr_func

        if save_results:
            return history(holder, mapper=apply_mapper_to_result)(curr_func)

        return curr_func

    return current_logged


def in_bounds(min_value: float, max_value: float):
    """TODO"""

    def current_in_bounds(func: Callable[P, float]) -> Callable[P, float]:
        @functools.wraps(func)
        def wrapper_apply(*args: P.args, **kwargs: P.kwargs) -> float:
            return min(max(func(*args, **kwargs), min_value), max_value)

        return wrapper_apply

    return current_in_bounds
