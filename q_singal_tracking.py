import numpy as np
import matplotlib.pyplot as plt
import control.matlab as ml

from data.LinearF16SS import A_long_hi as A, B_long_hi as B

# Define the state-space system
sys = ml.ss(A, B, np.eye(7), np.zeros((7, 2)))


# Define reference signal
def q_ref_step(t: np.ndarray, step_size: float, time_stamp: float) -> np.ndarray:
    q_ref = np.zeros_like(t)
    q_ref[t >= time_stamp] = step_size
    return q_ref

# Define controller
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    

# Initialize simulation parameters
dt = 0.01
t = np.arange(0, 1000 + dt, dt)
q_ref = q_ref_step(t, step_size=0.05, time_stamp=1.0)
x0 = np.zeros(7)

### Simulate the closed-loop system ###
x = x0.copy()
Controller = PIDController(Kp=4.0, Ki=1.0, Kd=1.5)
x_log = []
u_log = []

for i in range(len(t)):
    # Error signal
    q = x[4]
    error = q_ref[i] - q
    
    # PID controller
    delta_e = Controller.update(error, dt)
    u = np.array([0.0, delta_e])
    dxdt = sys.A @ x + sys.B @ u
    x += dxdt * dt
    
    x_log.append(x.copy())
    u_log.append(u.copy())

x_log = np.array(x_log)
u_log = np.array(u_log)
######################################

# Plot pitch rate tracking
plt.figure()
plt.plot(t, x_log[:, 4], label='q (pitch rate)')
plt.plot(t, q_ref, '--', label='q_ref (reference)')
plt.xlabel('Time (s)')
plt.ylabel('Pitch Rate (rad/s)')
plt.title('Pitch Rate Tracking')
plt.legend()
plt.grid()

# Plot elevator command
plt.figure()
plt.plot(t, np.rad2deg(u_log[:, 1]), label='Elevator deflection (deg)')
plt.xlabel('Time (s)')
plt.ylabel('Elevator (deg)')
plt.title('Control Input')
plt.legend()
plt.grid()
plt.show()







