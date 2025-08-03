"""
BlitzSchnell: A module for automatically optimizing parameters to improve performance.

This module provides utilities for automatically adjusting parameters like thread count,
batch size, etc., based on performance measurements.
"""

__all__ = [
    "OptimalParameter",
    "MultiLineSearchOptimizer",
    "Optimized",
]

import time
import math
import threading
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import signal
import random
from contextlib import contextmanager, AbstractContextManager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Generator,
)

from .optimized import Optimized

T = TypeVar("T")
R = TypeVar("R")


class OptimalParameter:
    """
    A class to optimize a numerical parameter based on performance measurements.
    """

    def __init__(
        self,
        initial_value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        noise_handling: str = "moving_average",
        noise_window: int = 5,
        exploration_factor: float = 0.2,
    ) -> None:
        self.value_: float = initial_value
        self.min_value: float = (
            min_value if min_value is not None else max(1, initial_value / 10)
        )
        self.max_value: float = (
            max_value if max_value is not None else initial_value * 10
        )
        self.history: List[Tuple[float, float]] = []  # [(value, performance), ...]
        self.start_time: Optional[float] = None

        # Noise handling
        self.noise_handling: str = noise_handling
        self.noise_window: int = max(2, noise_window)
        self.recent_performances: List[float] = []

        # For golden section search
        self.golden_ratio: float = (math.sqrt(5) + 1) / 2
        self.a: float = self.min_value
        self.b: float = self.max_value
        self.c: float = self.b - (self.b - self.a) / self.golden_ratio
        self.d: float = self.a + (self.b - self.a) / self.golden_ratio
        self.fc: Optional[float] = None
        self.fd: Optional[float] = None
        self.phase: int = 0  # 0: measure c, 1: measure d, 2: update a,b,c,d

        # Exploration factor (probability of trying a random value)
        self.exploration_factor: float = exploration_factor

        # Initial value is used for the first few measurements
        self.initial_value: float = initial_value
        self.measurement_count: int = 0
        self.warmup_count: int = 3

        # Best value found so far
        self.best_value: float = initial_value
        self.best_performance: float = float("-inf")

    def value(self) -> float:
        """Get the current optimal value of the parameter."""
        # During warmup, use the initial value
        if self.measurement_count < self.warmup_count:
            return self.initial_value

        # Occasionally try a random value to explore the parameter space
        if random.random() < self.exploration_factor:
            return random.uniform(self.min_value, self.max_value)

        # Use golden section search
        if self.phase == 0:
            return self.c
        elif self.phase == 1:
            return self.d

        return self.value_

    def start_measure(self) -> None:
        """Start measuring the performance."""
        self.start_time = time.time()

    def _handle_noise(self, performance: float) -> float:
        """Apply noise handling strategy to the raw performance measurement."""
        self.recent_performances.append(performance)

        # Keep only the most recent window of performances
        if len(self.recent_performances) > self.noise_window:
            self.recent_performances.pop(0)

        if self.noise_handling == "moving_average":
            # Simple moving average
            return sum(self.recent_performances) / len(self.recent_performances)
        elif self.noise_handling == "median":
            # Median filter (less sensitive to outliers)
            sorted_perfs = sorted(self.recent_performances)
            return sorted_perfs[len(sorted_perfs) // 2]
        elif self.noise_handling == "outlier_rejection":
            # Reject outliers (using mean ± 2*std_dev as threshold)
            if len(self.recent_performances) >= 3:
                mean = sum(self.recent_performances) / len(self.recent_performances)
                squared_diff_sum = sum(
                    (p - mean) ** 2 for p in self.recent_performances
                )
                std_dev = (squared_diff_sum / len(self.recent_performances)) ** 0.5

                # Filter out values outside 2 standard deviations
                filtered = [
                    p
                    for p in self.recent_performances
                    if mean - 2 * std_dev <= p <= mean + 2 * std_dev
                ]
                if filtered:
                    return sum(filtered) / len(filtered)
            # Fall back to moving average if we can't do outlier rejection
            return sum(self.recent_performances) / len(self.recent_performances)
        elif self.noise_handling == "exponential_smoothing":
            # Exponential smoothing (gives more weight to recent measurements)
            if len(self.recent_performances) == 1:
                return self.recent_performances[0]
            alpha = 0.3  # Smoothing factor
            result = self.recent_performances[0]
            for i in range(1, len(self.recent_performances)):
                result = alpha * self.recent_performances[i] + (1 - alpha) * result
            return result
        else:
            # No noise handling, return raw performance
            return performance

    def end_measure(self) -> float:
        """End measuring the performance and update the optimal value."""
        if self.start_time is None:
            raise ValueError("start_measure() must be called before end_measure()")
        elapsed_time: float = time.time() - self.start_time
        performance: float = 1 / elapsed_time  # Higher is better
        current_value: float = self.value()
        self.history.append((current_value, performance))

        # Apply noise handling before optimization
        filtered_performance: float = self._handle_noise(performance)

        # Update best value if this is better
        if filtered_performance > self.best_performance:
            self.best_performance = filtered_performance
            self.best_value = current_value

        self.measurement_count += 1

        # After warmup, start optimization
        if self.measurement_count >= self.warmup_count:
            self._optimize(current_value, filtered_performance)

        self.start_time = None
        return elapsed_time

    def signal_exception(self) -> None:
        """Signal an exception during measurement and adjust bounds.
        Assume failure means the current value is too high (common case for resource limits)
        If c failed, new interval is [a, c] -> update b
        If d failed, new interval is [a, d] -> update b
        If initial value failed, new interval is [a, initial_value] -> update b
        If exploration failed, new interval is [a, explored_value] -> update b
        """
        if self.start_time is None:
            raise ValueError("start_measure() must be called before signal_exception()")

        current_value = self.value()
        _original_bounds = (self.a, self.b)
        tolerance = 1e-9

        needs_bound_update = False
        if self.phase == 0 and abs(current_value - self.c) < tolerance:
            self.b = current_value
            needs_bound_update = True
            print(f"Exception at c ({current_value:.4f}). Setting upper bound.")
        elif self.phase == 1 and abs(current_value - self.d) < tolerance:
            self.b = current_value
            needs_bound_update = True
            print(f"Exception at d ({current_value:.4f}). Setting upper bound.")
        elif (
            self.measurement_count < self.warmup_count
            and abs(current_value - self.initial_value) < tolerance
        ):
            self.b = current_value
            needs_bound_update = True
            print(
                f"Exception at initial value ({current_value:.4f}). Setting upper bound."
            )
        elif random.random() < self.exploration_factor:
            if not (
                abs(current_value - self.c) < tolerance
                or abs(current_value - self.d) < tolerance
            ):
                self.b = current_value
                needs_bound_update = True

        if not needs_bound_update:
            self.b = current_value
            needs_bound_update = True

        self.a = max(self.min_value, self.a)
        self.b = min(self.max_value, self.b)

        if self.a >= self.b:
            if abs(self.a - self.b) < tolerance:
                nudge = tolerance * 10  # A small nudge value
                if self.b > self.min_value:
                    self.b = max(self.min_value, self.b - nudge)
                elif self.a < self.max_value:
                    self.a = max(self.min_value, self.a - nudge)

            if self.a >= self.b:
                raise ValueError(
                    f"Bounds have collapsed or inverted ({self.a} >= {self.b}) after exception at {current_value}; no valid range remains."
                )

        if self.b > self.a:
            self.c = self.b - (self.b - self.a) / self.golden_ratio
            self.d = self.a + (self.b - self.a) / self.golden_ratio
        else:
            self.c = self.a
            self.d = self.b

        self.fc = None
        self.fd = None
        self.phase = 0

        self.measurement_count += 1
        self.start_time = None

    def _optimize(self, current_value: float, performance: float) -> None:
        """Optimize the parameter value based on the measured performance."""
        # Golden Section Search for single parameter optimization
        if self.phase == 0:
            self.fc = performance
            self.phase = 1
        elif self.phase == 1:
            self.fd = performance
            self.phase = 2
            # Update a, b, c, d
            if (
                self.fc is not None and self.fc < self.fd
            ):  # We want to maximize performance
                self.a = self.c
                self.c = self.d
                self.fc = self.fd
                self.d = self.a + (self.b - self.a) / self.golden_ratio
                self.fd = None
            else:
                self.b = self.d
                self.d = self.c
                self.fd = self.fc
                self.c = self.b - (self.b - self.a) / self.golden_ratio
                self.fc = None
            self.phase = 0

        # Update the current best value
        self.value_ = (self.a + self.b) / 2

    def batched(self, items: Iterable[T]) -> Iterator[List[T]]:
        """Yield batches of items with the current optimal batch size."""
        items_list: List[T] = list(items)
        i: int = 0
        while i < len(items_list):
            batch_size: int = max(1, int(self.value()))
            yield items_list[i : i + batch_size]
            i += batch_size

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """A context manager for measuring performance."""
        self.start_measure()
        try:
            yield
        finally:
            self.end_measure()

    def get_history(self) -> List[Tuple[float, float]]:
        """Get the history of parameter values and their corresponding performances."""
        return self.history

    def get_best_value(self) -> float:
        """Get the best parameter value found so far."""
        if not self.history:
            return self.value_
        return self.best_value

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process."""
        if not self.history:
            return {
                "current_value": self.value_,
                "best_value": self.value_,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "measurements": 0,
                "performance_stats": None,
            }
        values: List[float] = [v for v, _ in self.history]
        performances: List[float] = [p for _, p in self.history]
        return {
            "current_value": self.value_,
            "best_value": self.best_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "measurements": len(self.history),
            "performance_stats": {
                "min": min(performances),
                "max": max(performances),
                "avg": sum(performances) / len(performances),
            },
        }

    def plot_history(self) -> bool:
        """Plot the optimization history if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            if not self.history:
                print("No optimization history available yet.")
                return False

            values: List[float] = [v for v, _ in self.history]
            performances: List[float] = [p for _, p in self.history]

            plt.figure(figsize=(10, 6))

            plt.subplot(2, 1, 1)
            plt.plot(values, marker="o")
            plt.title("Parameter Value Over Time")
            plt.xlabel("Measurement")
            plt.ylabel("Parameter Value")

            plt.subplot(2, 1, 2)
            plt.plot(performances, marker="x", color="red")
            plt.title("Performance Over Time")
            plt.xlabel("Measurement")
            plt.ylabel("Performance (1/time)")

            try:
                plt.savefig("optimization_history.png")
                print("Plot saved as 'optimization_history.png'")
            except Exception as e:
                print(f"Could not save plot: {e}")

            plt.tight_layout()
            plt.show()
            return True
        except ImportError:
            print(
                "Matplotlib is not available. Install it with 'pip install matplotlib' to use this feature."
            )
            return False


class MultiLineSearchOptimizer:
    """Optimize multiple parameters using coordinate descent with line search."""

    def __init__(
        self,
        parameter_configs: Dict[str, Dict[str, Union[float, None]]],
        noise_handling: str = "moving_average",
        noise_window: int = 5,
    ) -> None:
        self.optimal_parameters: Dict[str, OptimalParameter] = {}
        for name, config in parameter_configs.items():
            self.optimal_parameters[name] = OptimalParameter(
                initial_value=float(config.get("initial_value", 1.0) or 1.0),
                min_value=config.get("min_value", 0.1),
                max_value=config.get("max_value", 10.0),
                noise_handling=noise_handling,
                noise_window=noise_window,
            )
        self.param_names: List[str] = sorted(self.optimal_parameters.keys())
        self.current_param_index: int = 0
        self.start_time: Optional[float] = None
        self.history: List[Tuple[Dict[str, float], float]] = []
        self.performance_history: List[float] = []
        self.noise_handling: str = noise_handling
        self.noise_window: int = noise_window

    def values(self) -> Dict[str, float]:
        """Get the current optimal values for all parameters."""
        return {name: param.value() for name, param in self.optimal_parameters.items()}

    def start_measure(self) -> None:
        """Start measuring the performance."""
        current_param: str = self.param_names[self.current_param_index]
        self.optimal_parameters[current_param].start_measure()
        self.start_time = time.time()

    def end_measure(self) -> float:
        """End measuring the performance and update the optimization."""
        if self.start_time is None:
            raise ValueError("start_measure() must be called before end_measure()")
        elapsed_time: float = time.time() - self.start_time
        performance: float = 1 / elapsed_time
        current_values: Dict[str, float] = self.values()
        self.history.append((current_values.copy(), performance))
        current_param: str = self.param_names[self.current_param_index]
        self.optimal_parameters[current_param].end_measure()
        self.current_param_index = (self.current_param_index + 1) % len(
            self.param_names
        )
        self.start_time = None
        return elapsed_time

    def get_history(self) -> List[Tuple[Dict[str, float], float]]:
        """Get the history of parameter values and their performances."""
        return self.history

    def get_best_values(self) -> Dict[str, float]:
        """Get the best parameter values found so far."""
        return {
            name: param.get_best_value()
            for name, param in self.optimal_parameters.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process."""
        if not self.history:
            return {
                "current_values": self.values(),
                "best_values": self.values(),
                "measurements": 0,
                "performance_stats": None,
            }
        performances: List[float] = [p for _, p in self.history]
        return {
            "current_values": self.values(),
            "best_values": self.get_best_values(),
            "measurements": len(self.history),
            "performance_stats": {
                "min": min(performances),
                "max": max(performances),
                "avg": sum(performances) / len(performances),
                "latest": performances[-1],
            },
        }

    def plot_history(self) -> bool:
        """Plot the optimization history if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            if not self.history:
                print("No optimization history available yet.")
                return False

            param_histories: Dict[str, List[float]] = {
                name: [] for name in self.param_names
            }
            performances: List[float] = []
            for params, perf in self.history:
                performances.append(perf)
                for name in self.param_names:
                    param_histories[name].append(params[name])
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(performances, marker="o", linestyle="-")
            ax1.set_title("Performance Over Time")
            ax1.set_xlabel("Measurement")
            ax1.set_ylabel("Performance")
            ax2 = fig.add_subplot(2, 1, 2)
            for name in self.param_names:
                ax2.plot(param_histories[name], marker="x", linestyle="-", label=name)
            ax2.set_title("Parameter Values Over Time")
            ax2.set_xlabel("Measurement")
            ax2.set_ylabel("Parameter Value")
            ax2.legend()
            try:
                plt.savefig("multi_optimization_history.png")
                print("Plot saved as 'multi_optimization_history.png'")
            except Exception as e:
                print(f"Could not save plot: {e}")
            plt.tight_layout()
            plt.show()
            return True
        except ImportError:
            print(
                "Matplotlib is not available. Install it with 'pip install matplotlib' to use this feature."
            )
            return False
        except Exception as e:
            print(f"Error plotting history: {e}")
            return False
