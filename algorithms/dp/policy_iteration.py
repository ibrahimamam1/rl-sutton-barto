import numpy as np

def _policy_evaluation(env, policy, discount_factor, convergence_threshold):
    #initialiase values to 0
    V = np.zeros(env.observation_space.n)

    while(True):
        delta = 0

        for s in range(env.observation_space.n):
            if(s in env.terminal_states):
                continue 
            v = V[s]
            new_v = 0

            for action_idx, action in enumerate(env.action_space):
                action_prob = policy[s][action_idx]
                next_obs, reward, terminated, trans_prob = env.step(s, action)
                new_v += action_prob * trans_prob * (reward + discount_factor * V[next_obs])
            
            V[s] = new_v 
            delta = max(delta, abs(v-new_v))

        if(delta <= convergence_threshold):
            break

    return V 
                
def _policy_improvement(env, Values, discount_factor):
    new_policy = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        if s in env.terminal_states:
            continue

        #initialise action values for current state
        q_a = np.zeros(env.action_space.n)
        for action_idx, action in enumerate(env.action_space):
            next_obs, reward, terminated, trans_prob = env.step(s, action)
            q_a[action_idx] = trans_prob * (reward + discount_factor * Values[next_obs])
        
        greedy_action = np.argmax(q_a)
        new_policy[s][greedy_action] = 1

    return new_policy

def policy_iteration(env, discount_factor):
    #initialise convergence threshold
    theta = 1e-6

    #initialiase Values for states
    V = np.zeros(env.observation_space.n)

    #initialise policy to random policy
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    for i in range(env.observation_space.n):
        policy[i] = np.full(env.action_space.n, 1/env.action_space.n) #all actions have equal probability
    
    while(True):
        V_prime = _policy_evaluation(env, policy, discount_factor, theta)
        new_policy = _policy_improvement(env, V_prime, discount_factor)

        # Check if policy has converged
        if np.allclose(V_prime, V, atol=theta):
            break

        policy = new_policy
        V = V_prime
    
    return policy