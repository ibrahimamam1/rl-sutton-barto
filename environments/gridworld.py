import numpy as np
from .base_env import BaseEnv

class Gridworld(BaseEnv):
    """4x4 Gridworld from Sutton & Barto Example 4.1"""
    
    def __init__(self, rows=4, cols=4, terminal_states=None):
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states or [0, rows*cols-1]
        self.actions = ["up", "down", "left", "right"]
        self.action_deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
        self.current_state = None
        self.reset()
    
    def reset(self):
        """Reset to random non-terminal state"""
        while True:
            self.current_state = np.random.randint(0, self.rows * self.cols)
            if self.current_state not in self.terminal_states:
                break
        return self.current_state
    
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        if self.current_state in self.terminal_states:
            return self.current_state, 0, True, {}
        
        s = self._index_to_state(self.current_state)
        s_prime = self._next_state(s, action)
        next_state_idx = self._state_to_index(s_prime)
        
        reward = -1
        done = next_state_idx in self.terminal_states
        
        self.current_state = next_state_idx
        return next_state_idx, reward, done, {}
    
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
    
    def _state_to_index(self, state):
        return self.cols * state[0] + state[1]
    
    def _index_to_state(self, idx):
        return (idx // self.cols, idx % self.cols)
    
    @property
    def n_states(self):
        return self.rows * self.cols
    
    @property
    def n_actions(self):
        return len(self.actions)
    
    def get_transition_prob(self, state, action, next_state):
        """For model-based RL: return P(s'|s,a)"""
        s = self._index_to_state(state)
        expected_next = self._next_state(s, action)
        expected_idx = self._state_to_index(expected_next)
        return 1.0 if expected_idx == next_state else 0.0
    
    def get_reward(self, state, action, next_state):
        """For model-based RL: return R(s,a,s')"""
        if state in self.terminal_states:
            return 0
        return -1
