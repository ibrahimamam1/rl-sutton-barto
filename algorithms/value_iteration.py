import numpy as np
from environments.base_env import ModelBasedEnv
from collections import defaultdict

def value_iteration(env: ModelBasedEnv, policy, gamma=0.99, theta=0.0001):
    assert isinstance(env, ModelBasedEnv), "Value iteration is a dynamic programming algorithm and requires a model. use ModelBasedEnv"
    
    V = defaultdict(int)
    
    #1. Value iteration
    while True:
        delta = 0
        
        for state in range(env.n_states):
            v = V[state] 
            
            action_values = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                expected_reward = 0
                for next_state in range(env.n_states):
                    p, r = env.get_transition_model(state, action, next_state)
                    expected_reward += p * (r + gamma * V[next_state])
                action_values[action] = expected_reward
            
            V[state] = np.max(action_values)
            
            delta = max(delta, abs(V[state] - v))
            
        if delta < theta:
            break 

    # 2. Policy Extraction 
    for state in range(env.n_states):
        action_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            expected_reward = 0 
            for next_state in range(env.n_states):
                p, r = env.get_transition_model(state, action, next_state)
                expected_reward += p * (r + gamma * V[next_state])
            action_values[action] = expected_reward
            
        policy[state] = np.argmax(action_values)

    return V, policy 
