from gymnasium.spaces import Box, MultiDiscrete
import numpy as np


class Discretizer:
    def __init__(self, space: Box):
        self.lower_bounds = np.array(space.low, dtype=float)
        self.upper_bounds = np.array(space.high, dtype=float)


class UniformTileCoding(Discretizer):
    def __init__(self, space: Box, bins: tuple):
        super().__init__(space)
        self.bins = np.array(bins, dtype=int)
        self.fixed_dims = self.bins == 1  # Dimensions with only 1 bin are fixed

        # Avoid division by zero: bin width is arbitrary for fixed dims
        self.bin_widths = np.empty_like(self.lower_bounds)
        self.bin_widths[~self.fixed_dims] = (
            self.upper_bounds[~self.fixed_dims] - self.lower_bounds[~self.fixed_dims]
        ) / self.bins[~self.fixed_dims]
        self.bin_widths[self.fixed_dims] = 1.0  # dummy width for fixed dims

        self.space = MultiDiscrete(self.bins)

    def discretize(self, obs: tuple) -> tuple:
        obs = np.array(obs, dtype=float)
        obs = np.clip(obs, self.lower_bounds, self.upper_bounds - 1e-8)

        indices = np.zeros_like(obs, dtype=int)
        variable_dims = ~self.fixed_dims
        indices[variable_dims] = (
            (obs[variable_dims] - self.lower_bounds[variable_dims])
            / self.bin_widths[variable_dims]
        ).astype(int)

        # Fixed dims (bins=1) always map to index 0
        return tuple(indices)

    def undiscretize(self, indexes: tuple) -> tuple:
        indexes = np.array(indexes, dtype=int)
        values = self.lower_bounds + (indexes + 0.5) * self.bin_widths
        values[self.fixed_dims] = self.lower_bounds[self.fixed_dims]
        return tuple(values)
    
    def get_params(self):
        return {
            "bins": self.bins.tolist(),
            "low": self.lower_bounds.tolist(),
            "high": self.upper_bounds.tolist()
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
        self.space = MultiDiscrete(self.bins)


