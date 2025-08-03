# ⚡ BlitzSchnell

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

**⚡ BlitzSchnell** is a Python library for automatic performance optimization of common parameters like thread counts, batch sizes, chunk sizes, and timeouts. It eliminates the guesswork from performance tuning by automatically adapting these parameters at runtime based on measured performance.

## Why ⚡ BlitzSchnell?

Have you ever asked yourself:

- "What's the optimal thread count for my workload?"
- "What batch size will give the best performance?"
- "How big should my read buffer be?"

Instead of hardcoding these values or guessing, **⚡ BlitzSchnell** optimizes them dynamically as your code runs based on actual measured performance.

## Installation

⚡ BlitzSchnell is available on PyPI and can be installed using pip:

```bash
pip install blitzschnell
```

If you want Matplotlib support for visualization, you can install it with:

```bash
pip install blitzschnell[plotting]
```

## Core Concepts

⚡ BlitzSchnell uses **line search optimization** (golden section search) to automatically find optimal parameter values. It:

1. Measures the performance of your code with different parameter values
2. Adapts parameters to maximize performance
3. Handles noise in measurements with various filtering strategies
4. Continues to adapt as your workload changes

## Basic Usage

### The `Optimized` Class

The `Optimized` class is the easiest way to use BlitzSchnell. It takes a function and the name of the parameter you want to optimize, and it will automatically handle the optimization process for you.

```python
from blitzschnell import Optimized
import time

# A function that takes a parameter we want to optimize
def my_function(batch_size: int):
    # Simulate some work
    time.sleep(0.01 + 0.0001 * batch_size)

# Create an optimized version of the function
optimized_function = Optimized(
    func=my_function,
    param_name="batch_size",
    initial_value=1000,
    min_value=100,
    max_value=10000,
)

# Call the optimized function as you would the original
for _ in range(100):
    optimized_function()

# Get the best batch size found
summary = optimized_function.get_summary()
print(f"Optimal batch size: {summary['best_value']}")
```

### The `OptimalParameter` Class

For more control over the optimization process, you can use the `OptimalParameter` class. This class allows you to manually measure the performance of your code and update the optimization.

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
    with batch_size.measure():
        # Process the batch (simulate some work)
        time.sleep(0.01 + 0.0001 * len(batch))  # Example processing time
    
    i += size

print(f"Optimal batch size found: {batch_size.value()}")
```

## Advanced Usage

For more advanced use cases, such as multi-parameter optimization, see the [Advanced Usage](ADVANCED_USAGE.md) guide.

## How It Works

⚡ BlitzSchnell uses **golden section search** (a form of line search optimization) to efficiently find optimal parameter values by methodically narrowing down the search interval. For multiple parameters, it uses **coordinate descent**, optimizing one parameter at a time.

Key features:

1. **No external dependencies** - Works with standard library only
2. **Adaptive optimization** - Continues to adjust as workloads change
3. **Noise handling** - Multiple strategies to handle measurement noise
4. **Performance history** - Track how performance evolves

## License

MIT License - Free to use, modify, and distribute.