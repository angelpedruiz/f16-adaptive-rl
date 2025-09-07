import numpy as np
import matplotlib.pyplot as plt

class StepReferenceGenerator:
    """Standalone class to generate and plot cosine-smoothed step references."""
    
    def __init__(self, dt=0.01, max_time=30.0):
        self.dt = dt
        self.max_time = max_time
        self.max_steps = int(self.max_time / self.dt)
        
        self.cfg = {
            "amplitude": {
                "min": -20,
                "max": 20,
                "n_levels": 15,
            },
            "step_duration": {
                "min": 5.0,
                "max": 5.0,
                "n_levels": 1,
            },
            "transition_duration": {
                "min": 3.0,
                "max": 5.0,
                "n_levels": 5,
            }
        }
    
    @staticmethod
    def _cosine_step(x0, x1, y0, y1, dx):
        x = np.arange(x0, x1 + dx, dx)
        t = (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (1 - np.cos(np.pi * t)) / 2.0
        return x, y

    @staticmethod
    def _step_reference(x0, x1, y, dx):
        n_intervals = int(np.ceil((x1 - x0) / dx))
        x = np.linspace(x0, x1, n_intervals + 1)
        y_arr = np.full_like(x, fill_value=float(y), dtype=float)
        return x, y_arr

    def generate_cos_step_sequence(self, seed=None):
        rng = np.random.default_rng(seed)

        amplitude_levels = np.linspace(
            self.cfg["amplitude"]["min"],
            self.cfg["amplitude"]["max"],
            self.cfg["amplitude"]["n_levels"]
        )
        step_duration_levels = np.linspace(
            self.cfg["step_duration"]["min"],
            self.cfg["step_duration"]["max"],
            self.cfg["step_duration"]["n_levels"]
        )
        transition_duration_levels = np.linspace(
            self.cfg["transition_duration"]["min"],
            self.cfg["transition_duration"]["max"],
            self.cfg["transition_duration"]["n_levels"]
        )

        time = []
        reference = []

        current_t = 0.0
        current_y = 0.0

        while current_t < self.max_time:
            amp = rng.choice(amplitude_levels)
            trans_T = rng.choice(transition_duration_levels)
            step_T = rng.choice(step_duration_levels)

            # Transition
            t, r = self._cosine_step(current_t, current_t + trans_T, current_y, amp, self.dt)
            time.extend(t.tolist())
            reference.extend(r.tolist())
            current_t = t[-1]
            current_y = r[-1]

            if current_t >= self.max_time:
                break

            # Flat step
            t, r = self._step_reference(current_t, current_t + step_T, current_y, self.dt)
            time.extend(t.tolist())
            reference.extend(r.tolist())
            current_t = t[-1]
            current_y = r[-1]

        time = np.array(time[:self.max_steps])
        reference = np.array(reference[:self.max_steps])
        return time, reference


if __name__ == "__main__":
    gen = StepReferenceGenerator(dt=0.01, max_time=30.0)

    plt.figure(figsize=(10, 5))
    for i in range(10):
        t, ref = gen.generate_cos_step_sequence(seed=i)  # different seeds = different signals
        plt.plot(t, ref, label=f"Seq {i+1}")

    plt.xlabel("Time [s]")
    plt.ylabel("Reference")
    plt.title("10 Cosine-smoothed Step Reference Sequences")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
