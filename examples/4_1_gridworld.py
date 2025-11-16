import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from environments.gridworld import Gridworld
from algorithms.dp.policy_iteration import policy_iteration
from algorithms.dp.value_iteration import value_iteration
from utils.plotting import plot_value_function

import numpy as np
# Create environment
env = Gridworld(rows=4, cols=4)
#policy = policy_iteration(env, 0.95)
policy = value_iteration(env, 0.95, 1e-6)

