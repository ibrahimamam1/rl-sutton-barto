import numpy as np
from collections import defaultdict

def sarsa(env, n_iterations, alpha, gamma):
    Q = defaultdict(lambda: np.zeros(env.n_actions))

    print('Learning Values...')    
    for i in range(n_iterations): 
        obs,info = env.reset() 
        while True:
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[obs])

            next_obs, reward, done, info = env.step(action)
            action_prime = np.argmax(Q[next_obs])
            td_target = reward + gamma * Q[next_obs][action_prime] - Q[obs][action]
            td_error = Q[obs][action] - td_target
            Q[obs][action] += alpha * td_error
            
            if done:
                break
            obs = next_obs
    print('Values Learned.')
   
    print('Extracting Policy...')
    policy = defaultdict(lambda: np.zeros(env.n_actions))
    for s in env.get_states():
        policy[s] = np.argmax(Q[s]) 
    
    print('Policy Extracted')
    return Q, policy
            