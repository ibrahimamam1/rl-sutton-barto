import numpy as np
from tqdm import tqdm

def mc_policy_evaluation(env, policy, discount_factor, n_episodes):
    #initialize values to 0 for every state
    V = np.zeros(env.n_states)
    Returns = np.zeros(env.n_states)

    print('Evaluating Policy...')
    for i in tqdm(range(n_episodes)):
        #Generate an epsiode following the policy
        episode = []
        obs, info = env.reset()
        terminated = False 
        while not terminated:
            action_idx = np.argmax(policy[obs])
            new_obs, reward, terminated, truncated, info = env.step(action_idx)
            episode.append({"obs":obs, "action":action_idx, "reward":reward})
            obs = new_obs
        
        G = 0
        for step in episode:
            G = discount_factor*G + step['reward']
            Returns[step['obs']] = np.append(Returns[step['obs']], G)
            V[step['obs']] = np.average(Returns[step['obs']])
    
    print('Policy Evaluated')
    print(V)

