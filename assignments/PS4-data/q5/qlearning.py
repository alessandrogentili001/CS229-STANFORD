import numpy as np 
from plot_mountain_car import plot_mountain_car
from mountain_car import mountain_car
from IPython.display import clear_output
import time 

def qlearning(episodes, visualize=False):
    alpha = 0.05
    gamma = 0.99
    num_states = 100   # discretize state space 
    num_actions = 2
    actions = [-1, 1]  # Actions to move left (-1) and right (+1)

    #initialize q table 
    q = np.zeros((num_states, num_actions))
    steps_per_episode = np.zeros(episodes)

    # loop over episodes 
    for i in range(episodes):
        x, s, absorb = mountain_car([0.0, -np.pi / 6], 0) # starting at initial position 
        maxq = np.max(q[s, :])
        a = np.argmax(q[s, :])
        if q[s, 0] == q[s, 1]: a = np.random.randint(num_actions) # random choice 

        # loop until goal state is reached 
        steps = 0
        while not absorb:
            # Visualize the car's position
            if visualize:
                plot_mountain_car(x)
                clear_output(wait=True)
                time.sleep(0.01)

            # update car state after action 
            x, sn, absorb = mountain_car(x, actions[a])
            reward = -1 if not absorb else 0
            
            # update q value 
            q[s, a] = (1 - alpha) * q[s, a] + alpha * (reward + gamma * maxq)

            # choose next action 
            maxq = np.max(q[sn, :])
            an = np.argmax(q[sn, :])
            if q[sn, 0] == q[sn, 1]: an = np.random.randint(num_actions)

            # set parameters for the next iteration 
            a = an
            s = sn
            steps += 1

        steps_per_episode[i] = steps

    return q, steps_per_episode
