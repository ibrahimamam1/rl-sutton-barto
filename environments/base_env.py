# environments/base_env.py
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    """Base class for all RL environments"""
    
    @abstractmethod
    def reset(self):
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action):
        """Take action, return (next_state, reward, done, info)"""
        pass
    
    @property
    @abstractmethod
    def n_states(self):
        """Total number of states"""
        pass
    
    @property
    @abstractmethod
    def n_actions(self):
        """Total number of actions"""
        pass
