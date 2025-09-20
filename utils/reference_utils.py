import numpy as np


def cosine_smooth(x0, x1, y0, y1, dx):
    x = np.arange(x0, x1 + dx, dx)
    t = (x - x0) / (x1 - x0)
    y = y0 + (y1 - y0) * (1 - np.cos(np.pi * t)) / 2.0
    return x, y


def step_reference(x0, x1, y, dx):
    n_intervals = int(np.ceil((x1 - x0) / dx))
    x = np.linspace(x0, x1, n_intervals + 1)
    y_arr = np.full_like(x, fill_value=float(y), dtype=float)
    return x, y_arr


def generate_cos_step_sequence(cfg, max_time=30.0, dt=0.01, seed=None):
    max_steps = int(max_time / dt)
    rng = np.random.default_rng(seed)

    amplitude_levels = np.linspace(
        cfg["amplitude"]["min"], cfg["amplitude"]["max"], cfg["amplitude"]["n_levels"]
    )
    step_duration_levels = np.linspace(
        cfg["step_duration"]["min"],
        cfg["step_duration"]["max"],
        cfg["step_duration"]["n_levels"],
    )
    transition_duration_levels = np.linspace(
        cfg["transition_duration"]["min"],
        cfg["transition_duration"]["max"],
        cfg["transition_duration"]["n_levels"],
    )

    time = []
    reference = []

    current_t = 0.0
    current_y = 0.0

    while current_t < max_time:
        amp = rng.choice(amplitude_levels)
        trans_T = rng.choice(transition_duration_levels)
        step_T = rng.choice(step_duration_levels)

        # Transition
        t, r = cosine_smooth(current_t, current_t + trans_T, current_y, amp, dt)
        time.extend(t.tolist())
        reference.extend(r.tolist())
        current_t = t[-1]
        current_y = r[-1]

        if current_t >= max_time:
            break

        # Flat step
        t, r = step_reference(current_t, current_t + step_T, current_y, dt)
        time.extend(t.tolist())
        reference.extend(r.tolist())
        current_t = t[-1]
        current_y = r[-1]

    time = np.array(time[:max_steps])
    reference = np.array(reference[:max_steps])
    return time, reference


def generate_sin_sequence(cfg, max_time=30.0, dt=0.01):
    max_steps = int(max_time / dt)
    t = np.arange(max_steps) * dt
    period = cfg["T"]
    amplitude = cfg["A"]
    phase = cfg.get("phi", 0.0)
    y = amplitude * np.sin(2 * np.pi / period * t + phase)
    return t, y
