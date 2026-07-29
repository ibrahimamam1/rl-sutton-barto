import numpy as np
from environments.base_env import ModelFreeEnv
from collections import defaultdict

def q_learning(env:ModelFreeEnv, iters=500, gamma=0.9, lr=0.01, epsilon=0.05):
    Q = np.zeros((env.n_states, env.n_actions)) 
    policy = np.ones((env.n_states, env.n_actions), dtype=np.float32)/env.n_actions #init equally probable policy 

    for i in range(iters):
        obs = env.reset()
        action = np.random.choice(env.n_actions, p=policy[obs])
        done = False
        while not done:
            next_obs, reward, done = env.step(action)
            best_next_a = np.argmax(Q[next_obs])  # for the TD target only

            Q[obs, action] += lr * (reward + gamma*Q[next_obs, best_next_a] - Q[obs, action])

            best_a = np.argmax(Q[obs])
            policy[obs, :] = epsilon / env.n_actions
            policy[obs, best_a] += 1 - epsilon

            obs = next_obs
            action = np.random.choice(env.n_actions, p=policy[obs])  # sample behavior action fresh
    return Q, policy
