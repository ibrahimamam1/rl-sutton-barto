import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environments.blackjack import BlackJackEnv
from algorithms.mc.mc_prediction import mc_policy_evaluation
import numpy as np

env = BlackJackEnv()

policy = {}

# Define all possible states
for player_sum in range(2, 23):           # Player hand: 2 to 22
    for dealer_showing in range(2, 12):   # Dealer:2  to 11
        state = (player_sum, dealer_showing)
        if player_sum < 20:
            policy[state] = [1.0, 0.0]  # Always hit
        else:
            policy[state] = [0.0, 1.0]  # Always stand

# Evaluate the policy
V = mc_policy_evaluation(env, policy, discount_factor=1.0, n_episodes=10000)

#visualise the values 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

player_sums = range(2, 22)
dealer_cards = range(2, 12)

# Initialize value matrix
V_grid = np.zeros((len(player_sums), len(dealer_cards)))
V_grid[:] = np.nan  # Fill with NaN for missing states

# Fill in the values
for i, player in enumerate(player_sums):
    for j, dealer in enumerate(dealer_cards):
        state = (player, dealer)
        if state in V:
            V_grid[i, j] = V[state]

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(V_grid,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=dealer_cards,
            yticklabels=player_sums,
            cbar_kws={'label': 'State Value $V(s)$'},
            square=True,
            center=0.0)

plt.title("Blackjack State-Value Function $V(s)$\n(Policy: Hit if <20, else Stand)", fontsize=16)
plt.xlabel("Dealer's Showing Card")
plt.ylabel("Player's Sum")
plt.tight_layout()
plt.show()
