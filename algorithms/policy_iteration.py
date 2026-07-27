import numpy as np
from environments.base_env import ModelBasedEnv
from collections import defaultdict

def policy_iteration(env: ModelBasedEnv, policy, gamma=0.99):
   
    assert isinstance(env, ModelBasedEnv), "Policy iteration is a dynamic programming algorithm and requires a model. use ModelBasedEnv"
    V = defaultdict(int)
    while True:
        # 1. Policy Evaluation
        while True:
            delta = 0
            for state in range(env.n_states):
                v = V[state]
                action = policy[state]
                
                v_sum = 0
                for next_state in range(env.n_states):
                    p, r = env.get_transition_model(state, action, next_state)
                    v_sum += p * (r + gamma * V[next_state])
                
                V[state] = v_sum 
                delta = max(delta, abs(V[state] - v))
                
            if delta < 0.001:
                break 

        # 2. Policy Improvement
        policy_stable = True 
        q = np.zeros((env.n_states, env.n_actions))
        
        for state in range(env.n_states):
            old_action = policy[state]
            
            for action in range(env.n_actions):
                q_sum = 0 
                for next_state in range(env.n_states):
                    p, r = env.get_transition_model(state, action, next_state)
                    q_sum += p * (r + gamma * V[next_state])
                q[state][action] = q_sum 
                
            policy[state] = np.argmax(q[state])
            
            if old_action != policy[state]:
                policy_stable = False 

        if policy_stable:
            break
            
    return policy, V
