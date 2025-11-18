import numpy as np
from .base_env import BaseEnv

class Gridworld(BaseEnv):
    """4x4 Gridworld from Sutton & Barto Example 4.1"""
    
    def __init__(self, rows=4, cols=4, terminal_states=None):
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states or [0, rows*cols-1]
        self.actions_space = ["up", "down", "left", "right"]
        self.observation_space = np.arange(rows*cols) 
        self.action_to_idx = {a: i for i, a in enumerate(self.actions_space)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions_space)}
        self.action_deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
        self.reset()
   
    @property
    def n_states(self):
        return self.rows * self.cols
    
    @property
    def n_actions(self):
        return len(self.actions_space)
    
    
    def reset(self):
        """Reset to random non-terminal state"""
        while True:
            initial_state = np.random.randint(0, self.rows * self.cols)
            if initial_state not in self.terminal_states:
                break
        return initial_state
    
    def step(self, state, action):
        """Execute action and return (next_state, reward, transition_probability, done, info)"""
        if state in self.terminal_states:
            return state, 0, 1, True, {}
        s = self.get_state(state)
        s_prime = self._next_state(s, action)
        next_state_idx = self._state_to_index(s_prime)
        
        reward = -1
        done = next_state_idx in self.terminal_states
        
        return next_state_idx, reward, done, 1
    
    def get_actions(self, state=None):
        """Return all possible actions"""
        return self.actions.copy()
    
    def _next_state(self, state, action):
        """Calculate next state with boundary checking"""
        delta = self.action_deltas[action]
        next_s = (state[0] + delta[0], state[1] + delta[1])
        
        # Boundary check
        if (next_s[0] < 0 or next_s[0] >= self.rows or 
            next_s[1] < 0 or next_s[1] >= self.cols):
            return state  # Stay in place
        return next_s
    
      
