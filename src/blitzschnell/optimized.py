"""
A simplified, high-level interface for the BlitzSchnell optimization library.
"""

from . import OptimalParameter
from typing import Callable, Any, Dict, Optional

class Optimized:
    """
    A simplified interface for optimizing a function's parameter.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        param_name: str,
        initial_value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        noise_handling: str = "moving_average",
        noise_window: int = 5,
        exploration_factor: float = 0.2,
    ):
        self.func = func
        self.param_name = param_name
        self.optimizer = OptimalParameter(
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            noise_handling=noise_handling,
            noise_window=noise_window,
            exploration_factor=exploration_factor,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the function with the optimized parameter.
        """
        param_value = self.optimizer.value()
        kwargs[self.param_name] = param_value

        with self.optimizer.measure():
            return self.func(*args, **kwargs)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization process.
        """
        return self.optimizer.get_summary()

    def plot_history(self) -> bool:
        """
        Plot the optimization history.
        """
        return self.optimizer.plot_history()