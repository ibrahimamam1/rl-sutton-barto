import numpy as np

def value_iteration(env, discount_factor, convergence_threshold):

    #initialize V for all s to 0
    V = np.zeros(env.n_states)
    
    print("Iterating Values...")
    while(True):
        delta = 0
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue 

            v = V[s]
            q_a = np.zeros(env.n_actions)

            for action_idx, action in enumerate(env.actions_space):
                s_prime, reward, terminated, trans_prob = env.step(s, action)
                q = trans_prob*(reward + discount_factor*V[s_prime])
                q_a[action_idx] = q 

            V[s] = np.max(q_a)
            delta = max(delta, v-V[s])
        
        if(delta <= convergence_threshold):
            break 
    
    print("Values Converged:")
    print(V)
    
    #extract policy
    print('Extracting Optimal policy...')
    policy = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        if s in env.terminal_states:
            continue 

        q_a = np.zeros(env.n_actions)
        for action_idx, action in enumerate(env.actions_space):
                s_prime, reward, terminated, trans_prob = env.step(s, action)
                q = trans_prob*(reward + discount_factor*V[s_prime])
                q_a[action_idx] = q 
        optimal_action = np.argmax(q_a)
        policy[s][optimal_action] = 1
    
    print('Optimal Policy: ')
    print(policy)
    return policy

