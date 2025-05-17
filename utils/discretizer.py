import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.core import ObsType

class Discretizer:
    def __init__(self, space: Box):
        self.lower_bounds = np.array(space.low, dtype=float)
        self.upper_bounds = np.array(space.high, dtype=float)
        
class UniformTileCoding(Discretizer):
    def __init__(self, space: Box, bins: tuple):
        super().__init__(space)
        self.bins = np.array(bins, dtype=int)
        self.bin_widths = (self.upper_bounds - self.lower_bounds) / self.bins
        self.space = MultiDiscrete(self.bins)

    def discretize(self, obs: tuple) -> tuple:
        obs = np.array(obs, dtype=float)
        obs = np.clip(obs, self.lower_bounds, self.upper_bounds - 1e-8)
        indices = ((obs - self.lower_bounds) / self.bin_widths).astype(int)
        return tuple(indices)

    def undiscretize(self, indexes: tuple) -> tuple:
        indexes = np.array(indexes, dtype=int)
        continuous_values = self.lower_bounds + (indexes + 0.5) * self.bin_widths
        return tuple(continuous_values)


