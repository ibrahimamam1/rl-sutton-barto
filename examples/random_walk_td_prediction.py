import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from environments.random_walk import RandomWalkEnv
from algorithms.td_prediction import td_prediction
from algorithms.mc_prediction import mc_prediction

env = RandomWalkEnv()
policy = np.ones((env.n_states, env.n_actions), dtype=float) / env.n_actions

V1 = td_prediction(env, policy=policy, gamma=1)
V2 = mc_prediction(env, policy=policy, gamma=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot 1: TD estimates
ax1.plot(range(env.n_states), [V1[s] for s in range(env.n_states)], color='blue', label='TD')
ax1.set_xlabel("States")
ax1.set_ylabel("Value Estimates")
ax1.set_title("Value Function - TD Prediction")
ax1.legend()

# Plot 2: MC estimates
ax2.plot(range(env.n_states), [V2[s] for s in range(env.n_states)], color='red', label='MC')
ax2.set_xlabel("States")
ax2.set_ylabel("Value Estimates")
ax2.set_title("Value Function - MC Prediction")
ax2.legend()

plt.tight_layout()
plt.show()
