from gymnasium.spaces import Box, MultiDiscrete
import numpy as np


class Discretizer:
    def __init__(self, space: Box):
        self.lower_bounds = np.array(space.low, dtype=float)
        self.upper_bounds = np.array(space.high, dtype=float)


class UniformTileCoding(Discretizer):
    def __init__(self, space: Box, bins: tuple, return_tuple: bool = True):
        """
        Parameters
        ----------
        space : gymnasium.spaces.Box
            Continuous observation space.
        bins : tuple[int]
            Number of bins per dimension.
        return_tuple : bool, default=True
            Whether to return a tuple (hashable, slower) or a NumPy array (faster).
        """
        super().__init__(space)
        self.bins = np.array(bins, dtype=int)
        self.fixed_dims = self.bins == 1
        self.return_tuple = return_tuple

        # Compute bin widths
        self.bin_widths = np.empty_like(self.lower_bounds)
        self.bin_widths[~self.fixed_dims] = (
            (self.upper_bounds[~self.fixed_dims] - self.lower_bounds[~self.fixed_dims])
            / self.bins[~self.fixed_dims]
        )
        self.bin_widths[self.fixed_dims] = 1.0  # dummy width

        # Precompute inverse widths for faster multiply instead of divide
        self.inv_bin_widths = np.zeros_like(self.bin_widths)
        self.inv_bin_widths[~self.fixed_dims] = 1.0 / self.bin_widths[~self.fixed_dims]

        self.space = MultiDiscrete(self.bins)

    def discretize(self, obs) -> tuple | np.ndarray:
        """
        Map continuous observation to discrete indices.
        Accepts tuple, list, or ndarray as input.
        """
        # Ensure obs is a writable NumPy array
        obs_arr = np.asarray(obs, dtype=float)

        # Clip safely (always creates a new array if needed)
        obs_arr = np.clip(obs_arr, self.lower_bounds, self.upper_bounds - 1e-8)

        # Compute indices
        indices = ((obs_arr - self.lower_bounds) * self.inv_bin_widths).astype(int)

        # Force fixed dims to zero
        if np.any(self.fixed_dims):
            indices[self.fixed_dims] = 0

        return tuple(indices) if self.return_tuple else indices

    def undiscretize(self, indexes) -> tuple | np.ndarray:
        """
        Map discrete indices back to approximate continuous values.
        Accepts tuple, list, or ndarray as input.
        """
        indexes_arr = np.asarray(indexes, dtype=int)
        values = self.lower_bounds + (indexes_arr + 0.5) * self.bin_widths
        values[self.fixed_dims] = self.lower_bounds[self.fixed_dims]
        return tuple(values) if self.return_tuple else values

    def get_params(self):
        return {
            "bins": self.bins.tolist(),
            "low": self.lower_bounds.tolist(),
            "high": self.upper_bounds.tolist(),
        }

    def set_params(self, params):
        self.bins = np.array(params["bins"])
        self.lower_bounds = np.array(params["low"])
        self.upper_bounds = np.array(params["high"])
        self.fixed_dims = self.bins == 1

        self.bin_widths = np.empty_like(self.lower_bounds)
        self.bin_widths[~self.fixed_dims] = (
            self.upper_bounds[~self.fixed_dims] - self.lower_bounds[~self.fixed_dims]
        ) / self.bins[~self.fixed_dims]
        self.bin_widths[self.fixed_dims] = 1.0

        self.inv_bin_widths = np.zeros_like(self.bin_widths)
        self.inv_bin_widths[~self.fixed_dims] = 1.0 / self.bin_widths[~self.fixed_dims]

        self.space = MultiDiscrete(self.bins)
