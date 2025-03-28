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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import random
from contextlib import contextmanager


class OptimalParameter:
    """
    A class to optimize a numerical parameter based on performance measurements.
    """

    def __init__(
        self,
        initial_value,
        min_value=None,
        max_value=None,
        noise_handling="moving_average",
        noise_window=5,
        exploration_factor=0.2,
    ):
        self.value_ = initial_value
        self.min_value = (
            min_value if min_value is not None else max(1, initial_value / 10)
        )
        self.max_value = max_value if max_value is not None else initial_value * 10
        self.history = []  # [(value, performance), ...]
        self.start_time = None

        # Noise handling
        self.noise_handling = noise_handling
        self.noise_window = max(2, noise_window)
        self.recent_performances = []

        # For golden section search
        self.golden_ratio = (math.sqrt(5) + 1) / 2
        self.a = self.min_value
        self.b = self.max_value
        self.c = self.b - (self.b - self.a) / self.golden_ratio
        self.d = self.a + (self.b - self.a) / self.golden_ratio
        self.fc = None
        self.fd = None
        self.phase = 0  # 0: measure c, 1: measure d, 2: update a,b,c,d

        # Exploration factor (probability of trying a random value)
        self.exploration_factor = exploration_factor

        # Initial value is used for the first few measurements
        self.initial_value = initial_value
        self.measurement_count = 0
        self.warmup_count = 3

        # Best value found so far
        self.best_value = initial_value
        self.best_performance = float("-inf")

    def value(self):
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

    def start_measure(self):
        """Start measuring the performance."""
        self.start_time = time.time()

    def _handle_noise(self, performance):
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
            return sorted(self.recent_performances)[len(self.recent_performances) // 2]
        elif self.noise_handling == "outlier_rejection":
            # Reject outliers (using mean ± 2*std_dev as threshold)
            if (
                len(self.recent_performances) >= 3
            ):  # Need at least 3 points for meaningful stats
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

                if filtered:  # Make sure we didn't filter everything
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

    def end_measure(self):
        """End measuring the performance and update the optimal value."""
        if self.start_time is None:
            raise ValueError("start_measure() must be called before end_measure()")

        elapsed_time = time.time() - self.start_time
        performance = 1 / elapsed_time  # Higher is better
        current_value = self.value()
        self.history.append((current_value, performance))

        # Apply noise handling before optimization
        filtered_performance = self._handle_noise(performance)

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

    def _optimize(self, current_value, performance):
        """Optimize the parameter value based on the measured performance."""
        # Golden Section Search for single parameter optimization
        if self.phase == 0:
            self.fc = performance
            self.phase = 1
        elif self.phase == 1:
            self.fd = performance
            self.phase = 2
            # Update a, b, c, d
            if self.fc < self.fd:  # We want to maximize performance
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

    def batched(self, items):
        """Yield batches of items with the current optimal batch size."""
        i = 0
        while i < len(items):
            batch_size = max(1, int(self.value()))
            yield items[i : i + batch_size]
            i += batch_size

    @contextmanager
    def measure(self):
        """A context manager for measuring performance."""
        self.start_measure()
        try:
            yield
        finally:
            self.end_measure()

    def get_history(self):
        """Get the history of parameter values and their corresponding performances."""
        return self.history

    def get_best_value(self):
        """Get the best parameter value found so far."""
        if not self.history:
            return self.value_

        # Return the recorded best value
        return self.best_value

    def get_summary(self):
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

        values = [v for v, _ in self.history]
        performances = [p for _, p in self.history]

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

    def plot_history(self):
        """Plot the optimization history if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            if not self.history:
                print("No optimization history available yet.")
                return False

            values = [v for v, _ in self.history]
            performances = [p for _, p in self.history]

            # Create a new figure
            plt.figure(figsize=(10, 6))

            # Create subplots
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

            # Save the plot to a file first (as a backup)
            try:
                plt.savefig("optimization_history.png")
                print("Plot saved as 'optimization_history.png'")
            except Exception as e:
                print(f"Could not save plot: {e}")

            # Show the plot
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
        self, parameter_configs, noise_handling="moving_average", noise_window=5
    ):
        """
        Initialize the multi-parameter optimizer.

        Args:
            parameter_configs: Dictionary mapping parameter names to configurations.
                Each configuration should include: initial_value, min_value, max_value.
            noise_handling: Strategy for handling noise
            noise_window: Window size for noise handling strategies
        """
        self.parameters = {}
        self.optimal_parameters = {}

        # Create an OptimalParameter for each parameter to optimize
        for name, config in parameter_configs.items():
            self.optimal_parameters[name] = OptimalParameter(
                initial_value=config.get("initial_value", 1.0),
                min_value=config.get("min_value", 0.1),
                max_value=config.get("max_value", 10.0),
                noise_handling=noise_handling,
                noise_window=noise_window,
            )

        # For tracking which parameter to optimize next
        self.param_names = sorted(self.optimal_parameters.keys())
        self.current_param_index = 0

        # For tracking optimization progress
        self.start_time = None
        self.history = []  # [(params_dict, performance), ...]

        # Performance history for noise handling
        self.performance_history = []
        self.noise_handling = noise_handling
        self.noise_window = noise_window

    def values(self):
        """Get the current optimal values for all parameters."""
        return {name: param.value() for name, param in self.optimal_parameters.items()}

    def start_measure(self):
        """Start measuring the performance."""
        # Start measuring the current parameter being optimized
        current_param = self.param_names[self.current_param_index]
        self.optimal_parameters[current_param].start_measure()
        self.start_time = time.time()

    def end_measure(self):
        """End measuring the performance and update the optimization."""
        if self.start_time is None:
            raise ValueError("start_measure() must be called before end_measure()")

        elapsed_time = time.time() - self.start_time
        performance = 1 / elapsed_time  # Higher is better

        # Record the current values and performance
        current_values = self.values()
        self.history.append((current_values.copy(), performance))

        # Optimize the current parameter
        current_param = self.param_names[self.current_param_index]
        self.optimal_parameters[current_param].end_measure()

        # Move to the next parameter for the next measurement
        self.current_param_index = (self.current_param_index + 1) % len(
            self.param_names
        )

        self.start_time = None
        return elapsed_time

    def get_history(self):
        """Get the history of parameter values and their performances."""
        return self.history

    def get_best_values(self):
        """Get the best parameter values found so far."""
        return {
            name: param.get_best_value()
            for name, param in self.optimal_parameters.items()
        }

    def get_summary(self):
        """Get a summary of the optimization process."""
        if not self.history:
            return {
                "current_values": self.values(),
                "best_values": self.values(),
                "measurements": 0,
                "performance_stats": None,
            }

        performances = [p for _, p in self.history]

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

    def plot_history(self):
        """Plot the optimization history if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            if not self.history:
                print("No optimization history available yet.")
                return False

            # Extract data
            param_histories = {name: [] for name in self.param_names}
            performances = []

            for params, perf in self.history:
                performances.append(perf)
                for name in self.param_names:
                    param_histories[name].append(params[name])

            # Create figure with subplots
            fig = plt.figure(figsize=(12, 8))

            # Plot performance over time
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(performances, marker="o", linestyle="-")
            ax1.set_title("Performance Over Time")
            ax1.set_xlabel("Measurement")
            ax1.set_ylabel("Performance")

            # Plot parameter values over time
            ax2 = fig.add_subplot(2, 1, 2)

            for name in self.param_names:
                ax2.plot(param_histories[name], marker="x", linestyle="-", label=name)

            ax2.set_title("Parameter Values Over Time")
            ax2.set_xlabel("Measurement")
            ax2.set_ylabel("Parameter Value")
            ax2.legend()

            # Save the plot to a file
            try:
                plt.savefig("multi_optimization_history.png")
                print("Plot saved as 'multi_optimization_history.png'")
            except Exception as e:
                print(f"Could not save plot: {e}")

            # Show the plot
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
        initial_thread_count=None,
        min_threads=1,
        max_threads=None,
        noise_handling="moving_average",
    ):
        if initial_thread_count is None:
            initial_thread_count = multiprocessing.cpu_count()

        if max_threads is None:
            max_threads = multiprocessing.cpu_count() * 4

        self.thread_count = OptimalParameter(
            initial_thread_count,
            min_value=min_threads,
            max_value=max_threads,
            noise_handling=noise_handling,
        )
        self.executor = ThreadPoolExecutor(max_workers=int(self.thread_count.value()))
        self.lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        """Submit a task to the thread pool."""

        # Wrap the function to measure performance
        def wrapped_fn(*args, **kwargs):
            self.thread_count.start_measure()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                self.thread_count.end_measure()

                # Check if the thread count should be updated
                with self.lock:
                    current_thread_count = int(self.thread_count.value())
                    if current_thread_count != self.executor._max_workers:
                        # Create a new executor with the optimal thread count
                        old_executor = self.executor
                        self.executor = ThreadPoolExecutor(
                            max_workers=current_thread_count
                        )
                        # We don't wait for the old executor to shutdown to avoid deadlock
                        threading.Thread(
                            target=lambda: old_executor.shutdown(wait=True)
                        ).start()

        return self.executor.submit(wrapped_fn, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Map a function to each element in the iterables."""
        futures = [self.submit(fn, *args) for args in zip(*iterables)]

        if timeout is not None:
            end_time = time.time() + timeout
            for future in futures:
                remaining_time = max(0, end_time - time.time())
                try:
                    future.result(timeout=remaining_time)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")

        return [future.result() for future in futures]

    def shutdown(self, wait=True):
        """Shut down the thread pool."""
        self.executor.shutdown(wait=wait)

    def get_history(self):
        """Get the history of thread count values and their performances."""
        return self.thread_count.get_history()

    def get_summary(self):
        """Get a summary of the thread count optimization."""
        return self.thread_count.get_summary()

    def plot_history(self):
        """Plot the thread count optimization history."""
        return self.thread_count.plot_history()


class OptimalProcessPool:
    """
    A process pool with an optimal number of processes.
    """

    def __init__(
        self,
        initial_process_count=None,
        min_processes=1,
        max_processes=None,
        noise_handling="moving_average",
    ):
        if initial_process_count is None:
            initial_process_count = multiprocessing.cpu_count()

        if max_processes is None:
            max_processes = multiprocessing.cpu_count() * 2

        self.process_count = OptimalParameter(
            initial_process_count,
            min_value=min_processes,
            max_value=max_processes,
            noise_handling=noise_handling,
        )
        self.executor = ProcessPoolExecutor(max_workers=int(self.process_count.value()))
        self.lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        """Submit a task to the process pool and measure its performance."""
        start_time = time.time()
        future = self.executor.submit(fn, *args, **kwargs)

        # Add a callback to measure performance when the task is done
        def done_callback(f):
            try:
                # Retrieve the result to ensure any exceptions are raised
                f.result()

                elapsed_time = time.time() - start_time
                self.process_count.start_measure()
                self.process_count.end_measure()

                # Check if the process count should be updated
                with self.lock:
                    current_process_count = int(self.process_count.value())
                    if current_process_count != self.executor._max_workers:
                        # Create a new executor with the optimal process count
                        old_executor = self.executor
                        self.executor = ProcessPoolExecutor(
                            max_workers=current_process_count
                        )
                        # We don't wait for the old executor to shutdown to avoid deadlock
                        threading.Thread(
                            target=lambda: old_executor.shutdown(wait=True)
                        ).start()
            except Exception:
                # If the task raised an exception, we don't update the process count
                pass

        future.add_done_callback(done_callback)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Map a function to each element in the iterables."""
        futures = [self.submit(fn, *args) for args in zip(*iterables)]

        if timeout is not None:
            end_time = time.time() + timeout
            for future in futures:
                remaining_time = max(0, end_time - time.time())
                try:
                    future.result(timeout=remaining_time)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")

        return [future.result() for future in futures]

    def shutdown(self, wait=True):
        """Shut down the process pool."""
        self.executor.shutdown(wait=wait)

    def get_history(self):
        """Get the history of process count values and their performances."""
        return self.process_count.get_history()

    def get_summary(self):
        """Get a summary of the process count optimization."""
        return self.process_count.get_summary()

    def plot_history(self):
        """Plot the process count optimization history."""
        return self.process_count.plot_history()


class OptimalBatchProcessor:
    """
    Process items in optimal batches.
    """

    def __init__(
        self,
        initial_batch_size=1000,
        min_batch_size=1,
        max_batch_size=None,
        noise_handling="moving_average",
    ):
        self.batch_size = OptimalParameter(
            initial_batch_size,
            min_value=min_batch_size,
            max_value=max_batch_size,
            noise_handling=noise_handling,
        )

    def process(self, items, process_fn):
        """Process items in batches using the provided function."""
        results = []

        for batch in self.batch_size.batched(items):
            with self.batch_size.measure():
                batch_results = process_fn(batch)
                if batch_results:
                    results.extend(batch_results)

        return results

    def get_history(self):
        """Get the history of batch size values and their performances."""
        return self.batch_size.get_history()

    def get_summary(self):
        """Get a summary of the batch size optimization."""
        return self.batch_size.get_summary()

    def plot_history(self):
        """Plot the batch size optimization history."""
        return self.batch_size.plot_history()


class OptimalChunkProcessor:
    """
    Process chunks of items with optimal chunk size.
    """

    def __init__(
        self,
        initial_chunk_size=100,
        min_chunk_size=1,
        max_chunk_size=None,
        noise_handling="moving_average",
    ):
        self.chunk_size = OptimalParameter(
            initial_chunk_size,
            min_value=min_chunk_size,
            max_value=max_chunk_size,
            noise_handling=noise_handling,
        )

    def process(self, items, process_fn):
        """Process chunks of items using the provided function."""
        results = []

        i = 0
        while i < len(items):
            size = max(1, int(self.chunk_size.value()))
            chunk = items[i : i + size]

            with self.chunk_size.measure():
                result = process_fn(chunk)
                results.append(result)

            i += size

        return results

    def enumerate(self, items, process_fn):
        """Enumerate items and process them in optimal chunk sizes."""
        results = []

        i = 0
        while i < len(items):
            size = max(1, int(self.chunk_size.value()))
            chunk = [(j, items[j]) for j in range(i, min(i + size, len(items)))]

            with self.chunk_size.measure():
                chunk_results = [process_fn(idx, item) for idx, item in chunk]
                results.extend(chunk_results)

            i += size

        return results

    def get_history(self):
        """Get the history of chunk size values and their performances."""
        return self.chunk_size.get_history()

    def get_summary(self):
        """Get a summary of the chunk size optimization."""
        return self.chunk_size.get_summary()

    def plot_history(self):
        """Plot the chunk size optimization history."""
        return self.chunk_size.plot_history()


class OptimalFileReader:
    """
    Read files in optimal chunk sizes.
    """

    def __init__(
        self,
        initial_chunk_size=1024 * 1024,
        min_chunk_size=1024,
        max_chunk_size=None,
        noise_handling="moving_average",
    ):
        self.chunk_size = OptimalParameter(
            initial_chunk_size,
            min_value=min_chunk_size,
            max_value=max_chunk_size,
            noise_handling=noise_handling,
        )

    def read_file(self, file_path, process_chunk_fn=None):
        """Read a file in optimal chunk sizes."""
        if process_chunk_fn is None:
            # Generator mode
            with open(file_path, "rb") as f:
                while True:
                    with self.chunk_size.measure():
                        chunk = f.read(int(self.chunk_size.value()))

                    if not chunk:
                        break

                    yield chunk
        else:
            # Process mode
            results = []
            with open(file_path, "rb") as f:
                while True:
                    with self.chunk_size.measure():
                        chunk = f.read(int(self.chunk_size.value()))

                        if not chunk:
                            break

                        result = process_chunk_fn(chunk)
                        results.append(result)

            return results

    def get_history(self):
        """Get the history of chunk size values and their performances."""
        return self.chunk_size.get_history()

    def get_summary(self):
        """Get a summary of the chunk size optimization."""
        return self.chunk_size.get_summary()

    def plot_history(self):
        """Plot the chunk size optimization history."""
        return self.chunk_size.plot_history()


class HybridPool:
    """
    A hybrid pool that uses both threads and processes for optimal performance.
    """

    def __init__(
        self,
        initial_thread_count=None,
        initial_process_count=None,
        min_threads=1,
        max_threads=None,
        min_processes=1,
        max_processes=None,
        noise_handling="moving_average",
    ):
        self.thread_pool = OptimalThreadPool(
            initial_thread_count, min_threads, max_threads, noise_handling
        )
        self.process_pool = OptimalProcessPool(
            initial_process_count, min_processes, max_processes, noise_handling
        )

    def submit_cpu_bound(self, fn, *args, **kwargs):
        """Submit a CPU-bound task to the process pool."""
        return self.process_pool.submit(fn, *args, **kwargs)

    def submit_io_bound(self, fn, *args, **kwargs):
        """Submit an I/O-bound task to the thread pool."""
        return self.thread_pool.submit(fn, *args, **kwargs)

    def pipeline(self, items, cpu_bound_fn, io_bound_fn):
        """Process items in a pipeline: first CPU-bound, then I/O-bound."""
        cpu_futures = [self.submit_cpu_bound(cpu_bound_fn, item) for item in items]
        cpu_results = [future.result() for future in cpu_futures]

        io_futures = [
            self.submit_io_bound(io_bound_fn, result) for result in cpu_results
        ]
        io_results = [future.result() for future in io_futures]

        return io_results

    def shutdown(self, wait=True):
        """Shut down the hybrid pool."""
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)

    def get_thread_summary(self):
        """Get a summary of thread count optimization."""
        return self.thread_pool.get_summary()

    def get_process_summary(self):
        """Get a summary of process count optimization."""
        return self.process_pool.get_summary()


class OptimalBatchThreadPool:
    """A thread pool that optimizes both batch size and thread count together."""

    def __init__(
        self,
        initial_thread_count=None,
        initial_batch_size=100,
        min_threads=1,
        max_threads=None,
        min_batch_size=1,
        max_batch_size=10000,
        noise_handling="moving_average",
    ):
        if initial_thread_count is None:
            initial_thread_count = multiprocessing.cpu_count()

        if max_threads is None:
            max_threads = multiprocessing.cpu_count() * 4

        # Create multi-parameter optimizer for thread count and batch size
        self.optimizer = MultiLineSearchOptimizer(
            {
                "thread_count": {
                    "initial_value": initial_thread_count,
                    "min_value": min_threads,
                    "max_value": max_threads,
                },
                "batch_size": {
                    "initial_value": initial_batch_size,
                    "min_value": min_batch_size,
                    "max_value": max_batch_size,
                },
            },
            noise_handling=noise_handling,
        )

        # Initialize thread pool with initial count
        self.executor = ThreadPoolExecutor(max_workers=int(initial_thread_count))
        self.lock = threading.Lock()

    def process_in_batches(self, items, process_fn):
        """Process items in batches using the thread pool."""
        results = []
        i = 0

        while i < len(items):
            # Get current optimal values
            params = self.optimizer.values()
            thread_count = max(1, int(params["thread_count"]))
            batch_size = max(1, int(params["batch_size"]))

            # Update thread pool if needed
            with self.lock:
                if thread_count != self.executor._max_workers:
                    old_executor = self.executor
                    self.executor = ThreadPoolExecutor(max_workers=thread_count)
                    threading.Thread(
                        target=lambda: old_executor.shutdown(wait=True)
                    ).start()

            # Process a batch with the current optimal batch size
            batch_end = min(i + batch_size, len(items))
            batch = items[i:batch_end]

            # Prepare for measurement
            self.optimizer.start_measure()

            # Process batch items in parallel
            batch_results = list(self.executor.map(process_fn, batch))
            results.extend(batch_results)

            # End measurement and update optimizer
            self.optimizer.end_measure()

            i += batch_size

        return results

    def get_summary(self):
        """Get optimization summary."""
        return self.optimizer.get_summary()

    def get_history(self):
        """Get optimization history."""
        return self.optimizer.get_history()

    def plot_history(self):
        """Plot optimization history."""
        return self.optimizer.plot_history()

    def shutdown(self, wait=True):
        """Shut down the thread pool."""
        self.executor.shutdown(wait=wait)


def with_timeout(func, timeout, *args, **kwargs):
    """Execute a function with a timeout."""
    if not hasattr(signal, "SIGALRM"):
        # Windows doesn't have SIGALRM
        # Use a simple approach with a thread
        result = [None]
        exception = [None]

        def worker():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout} seconds"
            )

        if exception[0] is not None:
            raise exception[0]

        return result[0]
    else:
        # Unix-like systems with SIGALRM
        # Define a handler for the alarm signal
        def handler(signum, frame):
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout} seconds"
            )

        # Set up the alarm signal
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))

        try:
            result = func(*args, **kwargs)
        finally:
            # Reset the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        return result


class AdaptiveTimeout:
    """
    Execute operations with an adaptive timeout.
    """

    def __init__(
        self,
        initial_timeout=1.0,
        min_timeout=0.01,
        max_timeout=60.0,
        noise_handling="moving_average",
    ):
        self.timeout = OptimalParameter(
            initial_timeout,
            min_value=min_timeout,
            max_value=max_timeout,
            noise_handling=noise_handling,
        )

    def execute(self, func, *args, **kwargs):
        """Execute a function with the current optimal timeout."""
        self.timeout.start_measure()
        try:
            result = with_timeout(func, self.timeout.value(), *args, **kwargs)
            return result
        finally:
            self.timeout.end_measure()

    def get_history(self):
        """Get the history of timeout values and their performances."""
        return self.timeout.get_history()

    def get_summary(self):
        """Get a summary of the timeout optimization."""
        return self.timeout.get_summary()

    def plot_history(self):
        """Plot the timeout optimization history."""
        return self.timeout.plot_history()
