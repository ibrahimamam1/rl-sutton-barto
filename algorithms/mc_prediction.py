import numpy as np
from collections import defaultdict
from environments.base_env import ModelFreeEnv
from collections import defaultdict

def mc_prediction(env: ModelFreeEnv, policy, gamma=0.9, n_episodes=100):
    assert isinstance(env, ModelFreeEnv), "mc prediction is model-free; use ModelFreeEnv"

    V = defaultdict(int)
    returns = defaultdict(list)

    for ep in range(n_episodes):
        # Generate full episode following the policy
        states = [] 
        rewards = []

        obs = env.reset()
        done = False
        while not done:
            states.append(obs)
            action = policy[obs]
            obs, reward, done = env.step(action)
            rewards.append(reward)


        # Compute returns G_t by walking BACKWARD through the episode
        G = 0.0
        for t in reversed(range(len(states))):
            s_t = states[t]
            r_t_plus_1 = rewards[t]

            G = gamma * G + r_t_plus_1

            # First-visit check:
            if s_t not in states[:t]:
                returns[s_t].append(G)
                V[s_t] = np.mean(returns[s_t])

    return V
