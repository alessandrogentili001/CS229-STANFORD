import numpy as np

def mountain_car(x, a):
    '''
    INPUT
        x: current state of the car 
        a: action taken by the car 
    OUTPUT 
        x_next: next state of the car 
        s_idx: index of the state 
        absorb: reward 
    '''
    
    # Simulate forward
    x_next = np.zeros_like(x)                                # initialize new state
    x_next[1] = x[1] + 0.001 * a - 0.0025 * np.cos(3 * x[0]) # velocity (based on car position and gravity and the action)
    x_next[0] = x[0] + x_next[1]                             # position 

    # Clip the state to the bounds and check the goal state 
    absorb = 0
    if x_next[0] < -1.2: # bound reached 
        x_next[1] = 0 
    if x_next[0] > 0.5: # goal reached 
        absorb = 1
    x_next[0] = np.clip(x_next[0], -1.2, 0.5)   # clamps car position 
    x_next[1] = np.clip(x_next[1], -0.07, 0.07) # clamps car velocity 

    # Find the index of the state (after discretization)
    s_idx = int(10 * np.floor(10 * (x_next[0] + 1.2) / (1.7 + 1e-10)) + 
                np.floor(10 * (x_next[1] + 0.07) / (0.14 + 1e-10)) + 1)

    return x_next, s_idx, absorb
