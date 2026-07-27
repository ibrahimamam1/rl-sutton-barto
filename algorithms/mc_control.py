import numpy as np

def mc_control(env, gamma=0.9, epsilon=0.1, max_episodes=5000):
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions), dtype=int)
    
    # ε-greedy policy: uniform initially
    policy = np.ones((env.n_states, env.n_actions), dtype=float) / env.n_actions
    
    for ep in range(1, max_episodes + 1):
        # Generate episode using CURRENT policy
        episode = []          # list of (s, a, r)
        obs = env.reset()
        done = False
        while not done:
            # Sample action from epsilon-greedy policy for this state
            probs = policy[obs]
            action = np.random.choice(env.n_actions, p=probs)
            
            next_obs, reward, done = env.step(action)
            episode.append((obs, action, reward))
            obs = next_obs
        
        G = 0.0
        
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
        
        # Improve policy
        for s in range(env.n_states):
            best_a = np.argmax(Q[s])
            policy[s, :] = epsilon / env.n_actions
            policy[s, best_a] += (1.0 - epsilon)
    
    return Q, policy
