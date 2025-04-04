# BlitzSchnell

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

**BlitzSchnell** is a Python library for automatic performance optimization of common parameters like thread counts, batch sizes, chunk sizes, and timeouts. It eliminates the guesswork from performance tuning by automatically adapting these parameters at runtime based on measured performance.

## Why BlitzSchnell?

Have you ever asked yourself:

- "What's the optimal thread count for my workload?"
- "What batch size will give the best performance?"
- "How big should my read buffer be?"

Instead of hardcoding these values or guessing, **BlitzSchnell** optimizes them dynamically as your code runs based on actual measured performance.

## Installation

BlitzSchnell is available on PyPI and can be installed using pip:

```bash
pip install blitzschnell
```

If you want Matplotlib support for visualization, you can install it with:

```bash
pip install blitzschnell[plotting]
```

```python
# You can rename it to better fit your project
from blitzschnell import OptimalParameter, OptimalBatchProcessor, OptimalThreadPool
```

## Core Concepts

BlitzSchnell uses **line search optimization** (golden section search) to automatically find optimal parameter values. It:

1. Measures the performance of your code with different parameter values
2. Adapts parameters to maximize performance
3. Handles noise in measurements with various filtering strategies
4. Continues to adapt as your workload changes

## Examples

### 1. Basic Usage: Optimizing a Batch Size

```python
from blitzschnell import OptimalParameter
import time

# Create an optimizer for batch size
batch_size = OptimalParameter(initial_value=1000, min_value=100, max_value=10000)

items = list(range(100000))
i = 0

while i < len(items):
    # Get the current optimal batch size
    size = int(batch_size.value())
    
    # Get a batch of items
    batch = items[i:i+size]
    
    # Measure the performance of processing this batch
    batch_size.start_measure()
    
    # Process the batch (simulate some work)
    time.sleep(0.01 + 0.0001 * len(batch))  # Example processing time
    
    # End measurement and update the optimization
    batch_size.end_measure()
    
    i += size

print(f"Optimal batch size found: {batch_size.value()}")

# You can also get the optimization history and statistics
history = batch_size.get_history()
summary = batch_size.get_summary()
print(f"Best batch size: {summary['best_value']}")
```

### 2. Batch Processing with Automatic Optimization

```python
from blitzschnell import OptimalBatchProcessor
import time

# Create a batch processor
batch_processor = OptimalBatchProcessor(
    initial_batch_size=1000,
    min_batch_size=50,
    max_batch_size=5000
)

# Function to process a batch of items
def process_batch(batch):
    # Simulate work that depends on batch size
    time.sleep(0.01 + 0.0001 * len(batch))
    return [item * 2 for item in batch]

# Process all items with optimal batch sizes
items = list(range(10000))
results = batch_processor.process(items, process_batch)

print(f"Processed {len(results)} items")
print(f"Optimal batch size: {batch_processor.batch_size.value()}")
```

### 3. Optimal Thread Pool

```python
from blitzschnell import OptimalThreadPool
import time

# Create a thread pool with auto-optimized thread count
thread_pool = OptimalThreadPool(
    initial_thread_count=4,  # Start with 4 threads
    min_threads=1,
    max_threads=32
)

# Define a function to execute in parallel
def process_item(item):
    # Simulate some work
    time.sleep(0.1)
    return item * 2

try:
    # Process items in parallel with optimal thread count
    items = list(range(100))
    results = thread_pool.map(process_item, items)
    
    print(f"Processed {len(results)} items with {thread_pool.thread_count.value()} threads")
    
    # View optimization details
    summary = thread_pool.get_summary()
    print(f"Best thread count: {summary['best_value']}")
finally:
    thread_pool.shutdown()
```

### 4. Optimal Process Pool for CPU-Bound Tasks

```python
from blitzschnell import OptimalProcessPool
import time

# Create a process pool that automatically optimizes the process count
process_pool = OptimalProcessPool(
    min_processes=1,
    max_processes=16  # Adjust based on your machine
)

# Define a CPU-intensive function
def intensive_calculation(n):
    # Simulate a CPU-bound task
    result = 0
    for i in range(1000000):
        result += i * n
    return result

try:
    # Process items with optimal number of processes
    items = list(range(50))
    results = process_pool.map(intensive_calculation, items)
    
    print(f"Completed {len(results)} calculations")
    print(f"Optimal process count: {process_pool.process_count.value()}")
finally:
    process_pool.shutdown()
```

### 5. Optimal File Reading

```python
from blitzschnell import OptimalFileReader
import os

# Create a large test file
with open("large_file.txt", "w") as f:
    f.write("A" * 10000000)  # 10MB of data

# Create a file reader with auto-optimized chunk sizes
file_reader = OptimalFileReader(
    initial_chunk_size=64*1024,  # 64KB initial chunk size
    min_chunk_size=1024,         # 1KB minimum
    max_chunk_size=1024*1024*10  # 10MB maximum
)

# Option 1: Read file as a generator of optimally-sized chunks
total_bytes = 0
for chunk in file_reader.read_file("large_file.txt"):
    total_bytes += len(chunk)

# Option 2: Process chunks with a function
def count_bytes(chunk):
    return len(chunk)

chunk_sizes = file_reader.read_file("large_file.txt", count_bytes)
total_size = sum(chunk_sizes)

print(f"Read {total_size} bytes")
print(f"Optimal chunk size: {file_reader.chunk_size.value()} bytes")

# Clean up
os.remove("large_file.txt")
```

### 6. Chunk Processing with Optimal Chunk Sizes

```python
from blitzschnell import OptimalChunkProcessor
import time

# Create a chunk processor
chunk_processor = OptimalChunkProcessor(
    initial_chunk_size=100,
    min_chunk_size=10,
    max_chunk_size=1000
)

# Function to process chunks
def process_chunk(chunk):
    # Simulate work proportional to chunk size
    time.sleep(0.01 + 0.001 * len(chunk))
    return sum(chunk)

# Process items in optimally-sized chunks
items = list(range(10000))
results = chunk_processor.process(items, process_chunk)

print(f"Processed {len(items)} items in {len(results)} chunks")
print(f"Optimal chunk size: {chunk_processor.chunk_size.value()}")

# You can also process with index information
def process_with_index(idx, item):
    return idx, item * 2

indexed_results = chunk_processor.enumerate(items, process_with_index)
```

### 7. Hybrid Thread/Process Pool

```python
from blitzschnell import HybridPool
import time

# Create a hybrid pool that uses processes for CPU work and threads for I/O
hybrid_pool = HybridPool(
    initial_thread_count=8,
    initial_process_count=4
)

# Define CPU-bound and I/O-bound functions
def cpu_bound_task(item):
    # Simulate CPU-intensive work
    result = 0
    for i in range(1000000):
        result += i * item
    return result

def io_bound_task(item):
    # Simulate I/O work (like network or disk access)
    time.sleep(0.1)
    return f"Processed: {item}"

try:
    # Process items in a pipeline:
    # 1. CPU-bound processing in processes
    # 2. I/O-bound processing in threads
    items = list(range(20))
    results = hybrid_pool.pipeline(items, cpu_bound_task, io_bound_task)
    
    print(f"Processed {len(results)} items")
    print(f"Optimal process count: {hybrid_pool.process_pool.process_count.value()}")
    print(f"Optimal thread count: {hybrid_pool.thread_pool.thread_count.value()}")
finally:
    hybrid_pool.shutdown()
```

### 8. Multi-Parameter Optimization (Thread Count + Batch Size)

```python
from blitzschnell import OptimalBatchThreadPool
import time
import random

# Create a pool that optimizes both thread count and batch size together
batch_thread_pool = OptimalBatchThreadPool(
    initial_thread_count=4,
    initial_batch_size=100,
    min_threads=1,
    max_threads=16,
    min_batch_size=10,
    max_batch_size=1000,
    noise_handling='moving_average'
)

# Function to process an item
def process_item(item):
    # Simulate varying workload
    time.sleep(0.05 + random.random() * 0.05)
    return item * 2

try:
    # Process items with optimal thread count and batch size
    items = list(range(500))
    results = batch_thread_pool.process_in_batches(items, process_item)
    
    # Get optimization summary
    summary = batch_thread_pool.get_summary()
    best_values = summary['best_values']
    
    print(f"Processed {len(results)} items")
    print(f"Optimal thread count: {best_values['thread_count']:.1f}")
    print(f"Optimal batch size: {best_values['batch_size']:.1f}")
    
    # Plot the optimization history (if matplotlib is available)
    batch_thread_pool.plot_history()
finally:
    batch_thread_pool.shutdown()
```

### 9. Multiple Parameter Optimization

```python
from blitzschnell import MultiLineSearchOptimizer
import time
import threading
import random

# Create a multi-parameter optimizer
optimizer = MultiLineSearchOptimizer({
    'thread_count': {'initial_value': 4, 'min_value': 1, 'max_value': 16},
    'batch_size': {'initial_value': 500, 'min_value': 50, 'max_value': 5000},
    'timeout': {'initial_value': 1.0, 'min_value': 0.1, 'max_value': 5.0}
}, noise_handling='moving_average')

# Simulate a workload that depends on all parameters
def run_workload():
    params = optimizer.values()
    thread_count = int(params['thread_count'])
    batch_size = int(params['batch_size'])
    timeout = params['timeout']
    
    # Create and run threads
    threads = []
    for i in range(thread_count):
        t = threading.Thread(target=lambda: time.sleep(0.1 * random.random()))
        threads.append(t)
        t.start()
    
    # Process batches
    processing_time = 0.01 + (batch_size / 5000) * 0.2  # Simulate batch size impact
    time.sleep(processing_time)
    
    # Join threads with timeout
    for t in threads:
        t.join(timeout=min(timeout, 0.2))  # Cap actual timeout for example

# Run multiple iterations to optimize parameters
for i in range(20):
    optimizer.start_measure()
    run_workload()
    optimizer.end_measure()

# Get optimized parameters
best_values = optimizer.get_best_values()
print("\nOptimized Parameters:")
for param, value in best_values.items():
    print(f"  {param}: {value:.2f}")

# Plot the optimization history
optimizer.plot_history()
```

### 10. Adaptive Timeouts

```python
from blitzschnell import AdaptiveTimeout
import time
import random

# Create an adaptive timeout handler
timeout_handler = AdaptiveTimeout(
    initial_timeout=1.0,
    min_timeout=0.1,
    max_timeout=10.0,
    noise_handling='moving_average'
)

# Function that takes variable time to complete
def variable_duration_task(complexity):
    # Simulate a task that sometimes runs quickly, sometimes slowly
    duration = 0.2 + complexity * random.random() * 2
    time.sleep(duration)
    return f"Task completed with complexity {complexity}"

# Run tasks with adaptive timeouts
for i in range(20):
    complexity = random.uniform(0.1, 1.0)
    
    try:
        # Execute with adaptive timeout
        result = timeout_handler.execute(variable_duration_task, complexity)
        print(f"Run {i}: Success, timeout: {timeout_handler.timeout.value():.2f}s")
    except TimeoutError:
        print(f"Run {i}: Timed out after {timeout_handler.timeout.value():.2f}s")

# View optimization results
summary = timeout_handler.get_summary()
print(f"Optimal timeout: {summary['best_value']:.2f}s")
```

### 11. Using the Context Manager for Simple Performance Measurement

```python
from blitzschnell import OptimalParameter
import time

# Create an optimal parameter
chunk_size = OptimalParameter(1000, min_value=100, max_value=10000)

# Use the convenient context manager for measuring performance
for _ in range(10):
    # Get current optimal chunk size
    size = int(chunk_size.value())
    
    # Use context manager to measure and optimize
    with chunk_size.measure():
        # Simulate work that depends on chunk size
        time.sleep(0.01 + 0.001 * size**0.5)

print(f"Optimal chunk size: {chunk_size.value()}")
```

## Advanced Usage

### Handling Noisy Measurements

BlitzSchnell provides several strategies to handle noise in performance measurements:

```python
# Create a parameter with noise handling
batch_size = OptimalParameter(
    initial_value=1000,
    min_value=100,
    max_value=10000,
    noise_handling='moving_average',  # Options: 'moving_average', 'median', 'outlier_rejection', 'exponential_smoothing'
    noise_window=5  # Number of measurements to consider
)
```

### Visualizing Optimization Progress

You can visualize how parameters evolve over time:

```python
# After running your optimization
batch_processor.plot_history()  # Requires matplotlib
```

### Retrieving Optimization History and Statistics

```python
# Get the full optimization history
history = optimizer.get_history()  # List of (parameter_value, performance) tuples

# Get a summary of the optimization
summary = optimizer.get_summary()
print(f"Best value: {summary['best_value']}")
print(f"Current value: {summary['current_value']}")
print(f"Performance stats: {summary['performance_stats']}")
```

## How It Works

BlitzSchnell uses **golden section search** (a form of line search optimization) to efficiently find optimal parameter values by methodically narrowing down the search interval. For multiple parameters, it uses **coordinate descent**, optimizing one parameter at a time.

Key features:

1. **No external dependencies** - Works with standard library only
2. **Adaptive optimization** - Continues to adjust as workloads change
3. **Noise handling** - Multiple strategies to handle measurement noise
4. **Performance history** - Track how performance evolves

## License

MIT License - Free to use, modify, and distribute.
