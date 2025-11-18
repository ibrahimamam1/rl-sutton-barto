import numpy as np
from tqdm import tqdm
from collections import defaultdict

def mc_policy_evaluation(env, policy, discount_factor, n_episodes):
    
    V = {}
    Returns = defaultdict(list)

    print('Evaluating Policy...')
    for i in tqdm(range(n_episodes)):
        # Generate an epsiode following the policy
        episode = []
        obs, info = env.reset()
        terminated = False 
        while not terminated:
            action_idx = np.argmax(policy[obs])
            new_obs, reward, terminated, info = env.step(action_idx)
            episode.append({"obs": obs, "action": action_idx, "reward": reward})
            obs = new_obs
        
        G = 0
        for step in episode:
            G = discount_factor*G + step['reward']
            state = step['obs']
            Returns[state].append(G)
            V[state] = np.average(Returns[state])
    
    print('Policy Evaluated')
    return V

