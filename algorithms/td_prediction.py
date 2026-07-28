from collections import defaultdict
import numpy as np
from environments.base_env import ModelFreeEnv

def td_prediction(env: ModelFreeEnv, policy, gamma=0.9,lada=0, iters = 500, lr=0.01):
    V = defaultdict(int) 

    for i in range(iters):
        obs = env.reset() 
        done = False 

        while not done: 
            probs = policy[obs]
            action = np.random.choice(env.n_actions, p=probs) 
            next_obs, reward, done = env.step(action)
            V[obs] += lr*(reward + gamma*V[next_obs] - V[obs])
            obs = next_obs

    return V
