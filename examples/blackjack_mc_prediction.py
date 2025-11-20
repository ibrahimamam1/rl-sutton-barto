import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environments.blackjack import BlackJackEnv
from algorithms.mc.on_policy_mc_prediction import on_policy_mc_state_value_policy_evaluation_, on_policy_mc_state_action_value_policy_evaluation_
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
V = on_policy_mc_state_value_policy_evaluation_(env, policy, discount_factor=1.0, n_episodes=10000)
Q = on_policy_mc_state_action_value_policy_evaluation_(env, policy, discount_factor=1.0, epsilon=0.2, n_episodes=10000)

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


#visualise the Q values 
Q_hit_grid = np.zeros((len(player_sums), len(dealer_cards)))
Q_stand_grid = np.zeros((len(player_sums), len(dealer_cards)))
Q_hit_grid[:] = np.nan
Q_stand_grid[:] = np.nan

for i, player in enumerate(player_sums):
    for j, dealer in enumerate(dealer_cards):
        state = (player, dealer)
        if state in Q:
             # Q[state] is an array of size 2: [Q(s, hit), Q(s, stand)]
             # Assuming action 0 is Hit and 1 is Stand based on BlackJackEnv
             Q_hit_grid[i, j] = Q[state][0]
             Q_stand_grid[i, j] = Q[state][1]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot Q-values for Hit
sns.heatmap(Q_hit_grid,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=dealer_cards,
            yticklabels=player_sums,
            cbar_kws={'label': 'Q(s, Hit)'},
            square=True,
            center=0.0,
            ax=axes[0])
axes[0].set_title("Action-Value $Q(s, Hit)$", fontsize=16)
axes[0].set_xlabel("Dealer's Showing Card")
axes[0].set_ylabel("Player's Sum")

# Plot Q-values for Stand
sns.heatmap(Q_stand_grid,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=dealer_cards,
            yticklabels=player_sums,
            cbar_kws={'label': 'Q(s, Stand)'},
            square=True,
            center=0.0,
            ax=axes[1])
axes[1].set_title("Action-Value $Q(s, Stand)$", fontsize=16)
axes[1].set_xlabel("Dealer's Showing Card")
axes[1].set_ylabel("Player's Sum")

plt.tight_layout()
plt.show()
