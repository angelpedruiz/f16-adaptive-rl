# Standard action funtions for the system input => a(x, t)

def elevator_step(step, dt, step_size, time_stamp) -> tuple: 
    time = step * dt  
    if time < time_stamp:
        action = (0.0, 0.0)
    else:
        action = (0.0, step_size)   
    return action