import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from environments.gridworld import Gridworld
from algorithms.dp.policy_evaluation import policy_evaluation
from algorithms.dp.policy_improvement import policy_improvement
from utils.plotting import plot_value_function

import numpy as np
# Create environment
env = Gridworld(rows=4, cols=4)

# Define uniform random policy
policy = np.ones((env.n_states, env.n_actions)) * 0.25

# Run policy evaluation
V = policy_evaluation(env, policy, gamma=1.0, theta=1e-4)

# Run policy improvement
new_policy = policy_improvement(env, policy, V, gamma=1.0)

# Evaluate new policy
V_prime = policy_evaluation(env,new_policy,gamma=1.0, theta=1e-4)

print(f'old policy: {policy}')
print("Value Function:")
print(V.reshape(4, 4))
plot_value_function(V)

print(f'new policy: {new_policy}')
print("Value Function:")
print(V_prime.reshape(4, 4))
plot_value_function(V_prime)


