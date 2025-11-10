import numpy as np

def policy_evaluation(env, policy, gamma=1.0, theta=1e-4):
    """
    Evaluate a policy using iterative policy evaluation.
    
    Args:
        env: Environment with transition dynamics
        gamma: Discount factor
        theta: Convergence threshold
        policy: numpy array of shape (n_states, n_actions)
            policy[s][a] = probability of taking action a in state s
    """
    
    V = np.zeros(env.n_states)
    
    while True:
        delta = 0
        print('doing an iteration of policy evaluation')
        for s in range(env.n_states):
            print('in state:', s)
            if s in env.terminal_states:
                continue
            
            v = V[s]
            new_v = 0
            # Sum over all actions
            for action_idx, action in enumerate(env.actions):
                action_prob = policy[s][action_idx]  
                next_state, reward, trans_prob, done, info= env.step(s, action)
                new_v += action_prob * trans_prob * (reward + gamma * V[next_state])
            
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        
        if delta < theta:
            break
    
    return V
