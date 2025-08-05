from gymnasium.envs.registration import register

register(
    id="F16LinearModel-v0",
    entry_point="envs.f16_env:LinearModelF16",
)