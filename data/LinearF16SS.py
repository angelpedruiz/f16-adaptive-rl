import numpy as np
# State Space Representation of the F-16 High-Fidelity Longitudinal Model Linearised around trim conditions

# ==========================================================
#   Linearised Model For Longtiudinal Control High Fidelity
# ==========================================================

# State Matrix (7 x 7)
A_long_hi = np.array(
    [
        [0, 500.0000, 0.0000, -500.0000, 0, 0, 0],
        [0, 0, 0, 0, 1.0000, 0, 0],
        [0.0001, -32.1700, -0.0132, -2.6686, -1.1863, 0.0016, 0.0387],
        [0.0000, -0.0000, -0.0003, -0.6761, 0.9392, -0.0000, -0.0014],
        [0.0000, 0, -0.0000, -0.5757, -0.8741, 0, -0.1188],
        [0, 0, 0, 0, 0, -1.0000, 0],
        [0, 0, 0, 0, 0, 0, -20.2000],
    ]
)

# Input Matrix (7 x 2)
B_long_hi = np.array(
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1.0000, 0], [0, 20.2000]]
)

# Output Matrix
C = np.array(
    [
        [1.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 5.7296e01, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 1.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 5.7296e01, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 5.7296e01, 0.0000e00, 0.0000e00],
    ]
)

# Feedforward Matrix
D = np.zeros((5, 2))

# x = [Δh, Δθ, Δv, Δα, Δq, Δδ_t, Δδ_e]
# u = [Δδ_t, Δδ_e]
# units = [ft, rad, ft/s, rad, rad/s, rad, rad]
# trimmed_state = np.array([15000, ])

A_long_hi_ref = np.array(
    [
        [0.000e00, 5.000e02, 3.553e-10, -5.000e02, 0.000e00, 0.000e00, 0.000e00],
        [0.000e00, 0.000e00, 0.000e00, 0.000e00, 1.000e00, 0.000e00, 0.000e00],
        [1.074e-04, -3.217e01, -1.321e-02, -2.669e00, -1.186e00, 1.565e03, 3.870e-02],
        [
            2.076e-06,
            -3.681e-13,
            -2.552e-04,
            -6.761e-01,
            9.392e-01,
            -2.480e-07,
            -1.437e-03,
        ],
        [9.632e-12, 0.000e00, -1.184e-09, -5.757e-01, -8.741e-01, 0.000e00, -1.188e-01],
        [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, -1.000e00, 0.000e00],
        [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, -2.020e01],
    ]
)  # Reference state matrix from manual


eigsMine = np.linalg.eigvals(A_long_hi)  # To check stability of the system
eigsRef = np.linalg.eigvals(A_long_hi_ref)  # To check stability of the system

eigvals1 = np.linalg.eigvals(A_long_hi)
eigvals2 = np.linalg.eigvals(A_long_hi_ref)


# REFERNECE STATE MATRIX IS STABLE, I WILL USE THIS ONE

### TRACKING TASK ###


def theta_ref(t):
    """
    Reference pitch angle for tracking task. returns a sinusoidal signal in radians.
    """
    return 40 * np.pi / 180 * np.sin(2 * np.pi / 5 * t)  # Example reference signal


state_bounds = {
    "h": (-150, 400),  # Altitude in feet
    "theta": (-40 * np.pi / 180, 40 * np.pi / 180),  # Pitch angle in radians
    "v": (-40, 20),  # Velocity in ft/s
    "alpha": (-0.61, 0.6),  # Angle of attack in radians
    "q": (-1, 1),  # Pitch rate in rad/s
    "delta_t": (0, 0),  # Throttle in radians
    "delta_e": (-22, 1),  # Elevator deflection in radians
}  # state bounds achieved by the tuned PID attitude controler for theta_ref = 40[deg]*sin(2pi/5*t)

### FAULT SCENARIOS ###

# 1. Elevator effectiveness reduced by 70%

A_f1 = np.array(
    [
        [0, 499.999999990579, 0, -499.999999990579, 0, 0, 0],
        [0, 0, 0, 0, 1.0, 0, 0],
        [
            0.000109477186215166,
            -32.1699999993615,
            -0.0134596129665690,
            0.0905841154722147,
            -0.831692068584756,
            0.00156711106613637,
            -0.173799708014699,
        ],
        [
            2.07965627687335e-06,
            3.08710964758730e-13,
            -0.000255702521741442,
            -0.699409026570693,
            0.939568477182569,
            -1.90615966975822e-07,
            -0.00197058750029800,
        ],
        [
            1.73767933069683e-11,
            0,
            -2.13638037842034e-09,
            -1.61276821661576,
            -0.871992634697257,
            0,
            -0.0433079023586540,
        ],
        [0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, -20.2],
    ]
)
