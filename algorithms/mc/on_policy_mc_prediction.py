import numpy as np
from tqdm import tqdm
from collections import defaultdict

def on_policy_mc_state_value_policy_evaluation_(env, policy, discount_factor, n_episodes):
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

def on_policy_mc_state_action_value_policy_evaluation_(env, policy, discount_factor, epsilon, n_episodes):
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
            if np.random.uniform() < epsilon:
                action_idx = np.random.randint(env.n_actions)
            else:
                action_idx = np.argmax(policy[obs])

            new_obs, reward, terminated, info = env.step(action_idx)
            episode.append({"obs": obs, "action": action_idx, "reward": reward})
            obs = new_obs
        
        G = 0
        for step in episode:
            G = discount_factor*G + step['reward']
            state = step['obs']
            action = step['action']
            Returns[(state, action)].append(G)
            Q[state][action] = np.average(Returns[(state, action)])
    
    print('Policy Evaluated')
    return Q

