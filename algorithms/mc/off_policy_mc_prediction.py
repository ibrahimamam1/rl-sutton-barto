import numpy as np
from tqdm import tqdm
from collections import defaultdict

def off_policy_mc_state_value_policy_evaluation_(env, target_policy, behavior_policy, discount_factor, n_episodes):
    """
     Evaluates Values for a given policy using MC Prediction
    """

    V = {}
    Returns = defaultdict(list)

    print('Evaluating Policy...')
    for i in tqdm(range(n_episodes)):
        # Generate an epsiode following the policy
        episode = []
        obs, info = env.reset()
        terminated = False 
        while not terminated:
            action_idx = np.argmax(behavior_policy[obs])
            new_obs, reward, terminated, info = env.step(action_idx)
            episode.append({"obs": obs, "action": action_idx, "reward": reward})
            obs = new_obs
        
        G = 0
        for step in episode:
            G = discount_factor*G + step['reward']
            state = step['obs']
            importance_sampling_ratio = target_policy[state][step['action']] / behavior_policy[state][step['action']]
            Returns[state].append(G*importance_sampling_ratio)
            V[state] = np.average(Returns[state])
    
    print('Policy Evaluated')
    return V

def off_policy_mc_state_action_value_policy_evaluation_(env, target_policy, behavior_policy, discount_factor, n_episodes):
    """
     Evaluates State-Action Values for a given policy using MC Prediction
    """

    Q = defaultdict(lambda: np.zeros(env.n_actions))
    Returns = defaultdict(list)

    print('Evaluating Policy...')
    for i in tqdm(range(n_episodes)):
        # Generate an epsiode following the policy
        episode = []
        obs, info = env.reset()
        terminated = False 
        while not terminated:
            action_idx = np.argmax(behavior_policy[obs])
            new_obs, reward, terminated, info = env.step(action_idx)
            episode.append({"obs": obs, "action": action_idx, "reward": reward})
            obs = new_obs
        
        G = 0
        for step in episode:
            G = discount_factor*G + step['reward']
            state = step['obs']
            action = step['action']
            importance_sampling_ratio = target_policy[state][step['action']] / behavior_policy[state][step['action']]
            Returns[(state, action)].append(G*importance_sampling_ratio)
            Q[state][action] = np.average(Returns[(state, action)])
    
    print('Policy Evaluated')
    return Q

