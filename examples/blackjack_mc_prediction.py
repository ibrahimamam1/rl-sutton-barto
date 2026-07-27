import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environments.blackjack import BlackJackEnv
from algorithms.mc_prediction import mc_prediction
import numpy as np

env = BlackJackEnv()
policy = {}

# Define all possible states
for player_sum in range(2, 23):           # Player hand: 2 to 22
    for dealer_showing in range(2, 12):   # Dealer:2  to 11
        state = (player_sum, dealer_showing)
        if player_sum < 20:
            policy[state] = 1  # Always hit
        else:
            policy[state] = 0  # Always stand

# Evaluate the policy
V = mc_prediction(env, policy, gamma=1)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1.  Convert V into a grid
# ------------------------------------------------------------------
player_range = np.arange(2, 23)   # 2 .. 22
dealer_range = np.arange(2, 12)   # 2 .. 11 (dealer's showing card)

# Initialize with NaN so unvisited states are gray/blank
V_grid = np.full((len(dealer_range), len(player_range)), np.nan)

for (p, d), val in V.items():
    # Filter to the region we expect for this policy
    if 2 <= p <= 22 and 2 <= d <= 11:
        row = d - 2      # y-index (dealer)
        col = p - 2      # x-index (player)
        V_grid[row, col] = val

# ------------------------------------------------------------------
# 2.  Heatmap
# ------------------------------------------------------------------
plt.figure(figsize=(14, 6))
ax = sns.heatmap(
    V_grid,
    xticklabels=player_range,
    yticklabels=dealer_range,
    cmap="RdYlGn",          # diverging: red (bad) -> green (good)
    center=0,               # center colormap on 0
    vmin=-1, vmax=1,        # rewards are -1, 0, +1
    linewidths=0.5,
    cbar_kws={"label": "Estimated State Value V(s)"},
    annot=False             # set True if you want numbers inside cells
)
ax.set_xlabel("Player Hand Sum")
ax.set_ylabel("Dealer Showing Card")
ax.set_title("Value Function — Policy: Hit if Player < 20, Stand otherwise")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3.  (Optional) Line plot — easier to compare curves
# ------------------------------------------------------------------
plt.figure(figsize=(11, 5))
for d in dealer_range:
    values = [V.get((p, d), np.nan) for p in player_range]
    plt.plot(player_range, values, marker="o", label=f"Dealer {d}")

plt.axhline(0, color="black", linestyle="--", alpha=0.4)
plt.ylim(-1.05, 1.05)
plt.xlabel("Player Hand Sum")
plt.ylabel("Estimated Value V(s)")
plt.title("State Value by Dealer Showing Card")
plt.legend(title="Dealer", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
