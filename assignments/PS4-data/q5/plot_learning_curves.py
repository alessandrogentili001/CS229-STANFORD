import numpy as np
import matplotlib.pyplot as plt
from qlearning import qlearning


# Initialize an array to store episode steps for all runs
all_ep_steps = np.zeros((10, 10000))  

# Perform Q-learning 10 times
for i in range(10):
    q, ep_steps = qlearning(10000, visualize = False)  # Assume 10,000 episodes per run
    all_ep_steps[i, :] = ep_steps   # update steps array 

# Plot the mean number of steps over episodes
mean_ep_steps = np.mean(all_ep_steps, axis=0)  # Mean across the 10 runs
plt.plot(mean_ep_steps)
plt.xlabel('Number of Episodes')
plt.ylabel('Mean Steps per Episode')
plt.title('Q-learning Performance Over Episodes')
plt.show()
