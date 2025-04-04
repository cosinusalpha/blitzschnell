"""
BlitzSchnell: A module for automatically optimizing parameters to improve performance.

This module provides utilities for automatically adjusting parameters like thread count,
batch size, etc., based on performance measurements.
"""

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


class OptimalThreadPool:
    """
    A thread pool with an optimal number of threads.
    """

    def __init__(
        self,
        initial_thread_count: Optional[int] = None,
        min_threads: int = 1,
        max_threads: Optional[int] = None,
        noise_handling: str = "moving_average",
    ) -> None:
        if initial_thread_count is None:
            initial_thread_count = multiprocessing.cpu_count()
        if max_threads is None:
            max_threads = multiprocessing.cpu_count() * 4
        self.thread_count: OptimalParameter = OptimalParameter(
            initial_thread_count,
            min_value=min_threads,
            max_value=max_threads,
            noise_handling=noise_handling,
        )
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=int(self.thread_count.value())
        )
        self.lock: threading.Lock = threading.Lock()

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        """Submit a task to the thread pool."""

        def wrapped_fn(*args: Any, **kwargs: Any) -> R:
            self.thread_count.start_measure()
            try:
                result: R = fn(*args, **kwargs)
                return result
            finally:
                self.thread_count.end_measure()
                with self.lock:
                    current_thread_count: int = max(1, int(self.thread_count.value()))
                    if current_thread_count != self.executor._max_workers:
                        old_executor: ThreadPoolExecutor = self.executor
                        self.executor = ThreadPoolExecutor(
                            max_workers=current_thread_count
                        )
                        threading.Thread(
                            target=lambda: old_executor.shutdown(wait=True)
                        ).start()

        return self.executor.submit(wrapped_fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> List[R]:
        """Map a function to each element in the iterables."""
        futures: List[Future[R]] = [self.submit(fn, *args) for args in zip(*iterables)]
        if timeout is not None:
            end_time: float = time.time() + timeout
            for future in futures:
                remaining_time: float = max(0, end_time - time.time())
                try:
                    future.result(timeout=remaining_time)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")
        return [future.result() for future in futures]

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool."""
        self.executor.shutdown(wait=wait)

    def get_history(self) -> List[Tuple[float, float]]:
        """Get the history of thread count values and their performances."""
        return self.thread_count.get_history()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the thread count optimization."""
        return self.thread_count.get_summary()

    def plot_history(self) -> bool:
        """Plot the thread count optimization history."""
        return self.thread_count.plot_history()


class OptimalProcessPool:
    """
    A process pool with an optimal number of processes.
    """

    def __init__(
        self,
        initial_process_count: Optional[int] = None,
        min_processes: int = 1,
        max_processes: Optional[int] = None,
        noise_handling: str = "moving_average",
    ) -> None:
        if initial_process_count is None:
            initial_process_count = multiprocessing.cpu_count()
        if max_processes is None:
            max_processes = multiprocessing.cpu_count() * 2
        self.process_count: OptimalParameter = OptimalParameter(
            initial_process_count,
            min_value=min_processes,
            max_value=max_processes,
            noise_handling=noise_handling,
        )
        self.max_workers: int = int(self.process_count.value())
        self.executor: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=self.max_workers
        )
        self.lock: threading.Lock = threading.Lock()

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        """Submit a task to the process pool and measure its performance."""
        start_time: float = time.time()
        future: Future[R] = self.executor.submit(fn, *args, **kwargs)

        def done_callback(f: Future[R]) -> None:
            try:
                f.result()
                elapsed_time: float = time.time() - start_time
                self.process_count.start_measure()
                self.process_count.end_measure()
                with self.lock:
                    current_process_count = int(self.process_count.value())
                    if current_process_count != self.max_workers:
                        self.max_workers = current_process_count
                    if current_process_count != self.max_workers:
                        old_executor: ProcessPoolExecutor = self.executor
                        self.executor = ProcessPoolExecutor(
                            max_workers=current_process_count
                        )
                        threading.Thread(
                            target=lambda: old_executor.shutdown(wait=True)
                        ).start()
            except Exception:
                pass

        future.add_done_callback(done_callback)
        return future

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> List[R]:
        """Map a function to each element in the iterables."""
        futures: List[Future[R]] = [self.submit(fn, *args) for args in zip(*iterables)]
        if timeout is not None:
            end_time: float = time.time() + timeout
            for future in futures:
                remaining_time: float = max(0, end_time - time.time())
                try:
                    future.result(timeout=remaining_time)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")
        return [future.result() for future in futures]

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the process pool."""
        self.executor.shutdown(wait=wait)

    def get_history(self) -> List[Tuple[float, float]]:
        """Get the history of process count values and their performances."""
        return self.process_count.get_history()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the process count optimization."""
        return self.process_count.get_summary()

    def plot_history(self) -> bool:
        """Plot the process count optimization history."""
        return self.process_count.plot_history()


class OptimalBatchProcessor:
    """
    Process items in optimal batches.
    """

    def __init__(
        self,
        initial_batch_size: int = 1000,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        noise_handling: str = "moving_average",
    ) -> None:
        self.batch_size: OptimalParameter = OptimalParameter(
            initial_batch_size,
            min_value=min_batch_size,
            max_value=max_batch_size,
            noise_handling=noise_handling,
        )

    def process(
        self, items: Iterable[T], process_fn: Callable[[Iterable[T]], Iterable[Any]]
    ) -> List[Any]:
        """Process items in batches using the provided function."""
        results: List[Any] = []
        for batch in self.batch_size.batched(items):
            with self.batch_size.measure():
                batch_results = process_fn(batch)
                if batch_results:
                    results.extend(batch_results)
        return results

    def get_history(self) -> List[Tuple[float, float]]:
        """Get the history of batch size values and their performances."""
        return self.batch_size.get_history()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the batch size optimization."""
        return self.batch_size.get_summary()

    def plot_history(self) -> bool:
        """Plot the batch size optimization history."""
        return self.batch_size.plot_history()


class OptimalChunkProcessor:
    """
    Process chunks of items with optimal chunk size.
    """

    def __init__(
        self,
        initial_chunk_size: int = 100,
        min_chunk_size: int = 1,
        max_chunk_size: Optional[int] = None,
        noise_handling: str = "moving_average",
    ) -> None:
        self.chunk_size: OptimalParameter = OptimalParameter(
            initial_chunk_size,
            min_value=min_chunk_size,
            max_value=max_chunk_size,
            noise_handling=noise_handling,
        )

    def process(
        self, items: Sequence[T], process_fn: Callable[[Sequence[T]], Any]
    ) -> List[Any]:
        """Process chunks of items using the provided function."""
        results: List[Any] = []
        i: int = 0
        while i < len(items):
            size: int = max(1, int(self.chunk_size.value()))
            chunk: Sequence[T] = items[i : i + size]
            with self.chunk_size.measure():
                result = process_fn(chunk)
                results.append(result)
            i += size
        return results

    def enumerate(
        self, items: Sequence[T], process_fn: Callable[[int, T], Any]
    ) -> List[Any]:
        """Enumerate items and process them in optimal chunk sizes."""
        results: List[Any] = []
        i: int = 0
        while i < len(items):
            size: int = max(1, int(self.chunk_size.value()))
            chunk: List[Tuple[int, T]] = [
                (j, items[j]) for j in range(i, min(i + size, len(items)))
            ]
            with self.chunk_size.measure():
                chunk_results = [process_fn(idx, item) for idx, item in chunk]
                results.extend(chunk_results)
            i += size
        return results

    def get_history(self) -> List[Tuple[float, float]]:
        """Get the history of chunk size values and their performances."""
        return self.chunk_size.get_history()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the chunk size optimization."""
        return self.chunk_size.get_summary()

    def plot_history(self) -> bool:
        """Plot the chunk size optimization history."""
        return self.chunk_size.plot_history()


class OptimalFileReader:
    """
    Read files in optimal chunk sizes.
    """

    def __init__(
        self,
        initial_chunk_size: int = 1024 * 1024,
        min_chunk_size: int = 1024,
        max_chunk_size: Optional[int] = None,
        noise_handling: str = "moving_average",
    ) -> None:
        self.chunk_size: OptimalParameter = OptimalParameter(
            initial_chunk_size,
            min_value=min_chunk_size,
            max_value=max_chunk_size,
            noise_handling=noise_handling,
        )

    def read_file(
        self, file_path: str, process_chunk_fn: Optional[Callable[[bytes], Any]] = None
    ) -> Union[Iterator[bytes], List[Any]]:
        """Read a file in optimal chunk sizes."""
        if process_chunk_fn is None:

            def generator() -> Iterator[bytes]:
                with open(file_path, "rb") as f:
                    while True:
                        with self.chunk_size.measure():
                            chunk = f.read(int(self.chunk_size.value()))
                        if not chunk:
                            break
                        yield chunk

            return generator()
        else:
            results: List[Any] = []
            with open(file_path, "rb") as f:
                while True:
                    with self.chunk_size.measure():
                        chunk = f.read(int(self.chunk_size.value()))
                        if not chunk:
                            break
                        result = process_chunk_fn(chunk)
                        results.append(result)
            return results

    def get_history(self) -> List[Tuple[float, float]]:
        """Get the history of chunk size values and their performances."""
        return self.chunk_size.get_history()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the chunk size optimization."""
        return self.chunk_size.get_summary()

    def plot_history(self) -> bool:
        """Plot the chunk size optimization history."""
        return self.chunk_size.plot_history()


class HybridPool:
    """
    A hybrid pool that uses both threads and processes for optimal performance.
    """

    def __init__(
        self,
        initial_thread_count: Optional[int] = None,
        initial_process_count: Optional[int] = None,
        min_threads: int = 1,
        max_threads: Optional[int] = None,
        min_processes: int = 1,
        max_processes: Optional[int] = None,
        noise_handling: str = "moving_average",
    ) -> None:
        self.thread_pool: OptimalThreadPool = OptimalThreadPool(
            initial_thread_count, min_threads, max_threads, noise_handling
        )
        self.process_pool: OptimalProcessPool = OptimalProcessPool(
            initial_process_count, min_processes, max_processes, noise_handling
        )

    def submit_cpu_bound(
        self, fn: Callable[..., R], *args: Any, **kwargs: Any
    ) -> Future[R]:
        """Submit a CPU-bound task to the process pool."""
        return self.process_pool.submit(fn, *args, **kwargs)

    def submit_io_bound(
        self, fn: Callable[..., R], *args: Any, **kwargs: Any
    ) -> Future[R]:
        """Submit an I/O-bound task to the thread pool."""
        return self.thread_pool.submit(fn, *args, **kwargs)

    def pipeline(
        self,
        items: Iterable[T],
        cpu_bound_fn: Callable[[T], R],
        io_bound_fn: Callable[[R], R],
    ) -> List[R]:
        """Process items in a pipeline: first CPU-bound, then I/O-bound."""
        cpu_futures: List[Future[R]] = [
            self.submit_cpu_bound(cpu_bound_fn, item) for item in items
        ]
        cpu_results: List[R] = [future.result() for future in cpu_futures]
        io_futures: List[Future[R]] = [
            self.submit_io_bound(io_bound_fn, result) for result in cpu_results
        ]
        io_results: List[R] = [future.result() for future in io_futures]
        return io_results

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the hybrid pool."""
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)

    def get_thread_summary(self) -> Dict[str, Any]:
        """Get a summary of thread count optimization."""
        return self.thread_pool.get_summary()

    def get_process_summary(self) -> Dict[str, Any]:
        """Get a summary of process count optimization."""
        return self.process_pool.get_summary()


class OptimalBatchThreadPool:
    """A thread pool that optimizes both batch size and thread count together."""

    def __init__(
        self,
        initial_thread_count: Optional[int] = None,
        initial_batch_size: int = 100,
        min_threads: int = 1,
        max_threads: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 10000,
        noise_handling: str = "moving_average",
    ) -> None:
        if initial_thread_count is None:
            initial_thread_count = multiprocessing.cpu_count()
        if max_threads is None:
            max_threads = multiprocessing.cpu_count() * 4
        self.optimizer: MultiLineSearchOptimizer = MultiLineSearchOptimizer(
            {
                "thread_count": {
                    "initial_value": float(initial_thread_count),
                    "min_value": float(min_threads),
                    "max_value": float(max_threads),
                },
                "batch_size": {
                    "initial_value": float(initial_batch_size),
                    "min_value": float(min_batch_size),
                    "max_value": float(max_batch_size),
                },
            },
            noise_handling=noise_handling,
        )
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=int(initial_thread_count)
        )
        self.lock: threading.Lock = threading.Lock()

    def process_in_batches(
        self, items: Sequence[T], process_fn: Callable[[T], R]
    ) -> List[R]:
        """Process items in batches using the thread pool."""
        results: List[R] = []
        i: int = 0
        while i < len(items):
            params: Dict[str, float] = self.optimizer.values()
            thread_count: int = max(1, int(params["thread_count"]))
            batch_size: int = max(1, int(params["batch_size"]))
            with self.lock:
                if thread_count != self.executor._max_workers:
                    old_executor: ThreadPoolExecutor = self.executor
                    self.executor = ThreadPoolExecutor(max_workers=thread_count)
                    threading.Thread(
                        target=lambda: old_executor.shutdown(wait=True)
                    ).start()
            batch_end: int = min(i + batch_size, len(items))
            batch: Sequence[T] = items[i:batch_end]
            self.optimizer.start_measure()
            batch_results: List[R] = list(self.executor.map(process_fn, batch))
            results.extend(batch_results)
            self.optimizer.end_measure()
            i += batch_size
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return self.optimizer.get_summary()

    def get_history(self) -> List[Tuple[Dict[str, float], float]]:
        """Get optimization history."""
        return self.optimizer.get_history()

    def plot_history(self) -> bool:
        """Plot optimization history."""
        return self.optimizer.plot_history()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool."""
        self.executor.shutdown(wait=wait)


def with_timeout(
    func: Callable[..., R], timeout: float, *args: Any, **kwargs: Any
) -> R:
    """Execute a function with a timeout."""
    if not hasattr(signal, "SIGALRM"):
        result: Optional[R] = None
        exception: Optional[Exception] = None

        def worker() -> None:
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout} seconds"
            )
        if exception is not None:
            raise exception
        if result is None:
            raise ValueError("Function did not return a result.")
        return result
    else:

        def handler(signum: int, frame: Any) -> None:
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout} seconds"
            )

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result


class AdaptiveTimeout:
    """
    Execute operations with an adaptive timeout.
    """

    def __init__(
        self,
        initial_timeout: float = 1.0,
        min_timeout: float = 0.01,
        max_timeout: float = 60.0,
        noise_handling: str = "moving_average",
    ) -> None:
        self.timeout: OptimalParameter = OptimalParameter(
            initial_timeout,
            min_value=min_timeout,
            max_value=max_timeout,
            noise_handling=noise_handling,
        )

    def execute(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Execute a function with the current optimal timeout."""
        self.timeout.start_measure()
        try:
            result: R = with_timeout(func, self.timeout.value(), *args, **kwargs)
            return result
        finally:
            self.timeout.end_measure()

    def get_history(self) -> List[Tuple[float, float]]:
        """Get the history of timeout values and their performances."""
        return self.timeout.get_history()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the timeout optimization."""
        return self.timeout.get_summary()

    def plot_history(self) -> bool:
        """Plot the timeout optimization history."""
        return self.timeout.plot_history()
