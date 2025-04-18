from blitzschnell import OptimalParameter
import time


def opt_function(x: int) -> None:
    """Test Function with a known minimum at x=300."""
    m = 0.0001
    zero = 300
    y = m * abs(x - zero)
    time.sleep(y)  # Simulate some work


# Create an optimizer for batch size
batch_size = OptimalParameter(
    initial_value=5000,
    min_value=0,
    max_value=10000,
    noise_handling="none",  # Disable noise handling for this example
    exploration_factor=0,  # Disable exploration for this example
)

max_iter = 100

for i in range(max_iter):
    size = int(batch_size.value())  # Get optimized batch size
    print(f"Size: {size}")

    batch_size.start_measure()  # Start performance measurement
    opt_function(size)
    data = batch_size.end_measure()  # End performance measurement

print(f"Optimal batch size found: {batch_size.value()}")
