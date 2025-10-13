import matplotlib.pyplot as plt
import numpy as np

def plot_value_function(V, shape=(4, 4), title="Value Function"):
    """Plot value function as heatmap"""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(V.reshape(shape), cmap='coolwarm', interpolation='nearest')
    
    # Add values as text
    for i in range(shape[0]):
        for j in range(shape[1]):
            text = ax.text(j, i, f'{V[i*shape[1] + j]:.1f}',
                          ha="center", va="center", color="black")
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

def plot_policy(policy, shape=(4, 4), title="Policy"):
    """Plot policy as arrows on grid"""
    arrow_map = {
        "up": "↑",
        "down": "↓", 
        "left": "←",
        "right": "→"
    }
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = i * shape[1] + j
            action = policy.get(state, "")
            if action:
                ax.text(j, i, arrow_map.get(action, action),
                       ha="center", va="center", fontsize=20)
    
    ax.set_xlim(-0.5, shape[1]-0.5)
    ax.set_ylim(-0.5, shape[0]-0.5)
    ax.set_xticks(range(shape[1]))
    ax.set_yticks(range(shape[0]))
    ax.grid(True)
    ax.set_title(title)
    ax.invert_yaxis()
    return fig
