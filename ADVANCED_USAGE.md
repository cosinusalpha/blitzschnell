# Advanced Usage

This guide covers advanced use cases for the ⚡ BlitzSchnell library.

## Multi-Parameter Optimization

You can optimize multiple parameters at once using the `MultiLineSearchOptimizer` class.

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

## Other Advanced Examples

For more advanced examples, including `OptimalThreadPool`, `OptimalProcessPool`, `OptimalBatchProcessor`, `OptimalChunkProcessor`, `OptimalFileReader`, `HybridPool`, `OptimalBatchThreadPool`, and `AdaptiveTimeout`, please see the original `README.md` file in the [git repository](https://github.com/your-username/blitzschnell/blob/main/README.md).

```