import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from environments.gridworld import Gridworld
from algorithms.dp.policy_evaluation import policy_evaluation
from utils.plotting import plot_value_function

# Create environment
env = Gridworld(rows=4, cols=4)

# Define uniform random policy
policy = {}
for s in range(env.n_states):
    for a in env.get_actions():
        policy[(s, a)] = 0.25

# Run policy evaluation
V = policy_evaluation(env, policy, gamma=1.0, theta=1e-4)

# Visualize
print("Value Function:")
print(V.reshape(4, 4))
plot_value_function(V)
