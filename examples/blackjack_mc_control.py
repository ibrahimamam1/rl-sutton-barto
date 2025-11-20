import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.mc.on_policy_mc_control import on_policy_mc_control
from environments.blackjack import BlackJackEnv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

env = BlackJackEnv()
policy = on_policy_mc_control(env, n_iterations=500000, epsilon=0.2)
V = on_policy_mc_state_value_policy_evaluation_(env, policy, discount_factor=1, epsilon=0.2, n_episodes=5000)
    

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


