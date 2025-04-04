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
    batch = items[i : i + size]

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
