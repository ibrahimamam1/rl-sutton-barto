import numpy as np

def policy_evaluation(env, policy, gamma=1.0, theta=1e-4):
    """
    Evaluate a policy using iterative policy evaluation.
    
    Args:
        env: Environment with transition dynamics
        policy: Dict mapping state -> action probabilities
        gamma: Discount factor
        theta: Convergence threshold
    
    Returns:
        V: Value function array
    """
    V = np.zeros(env.n_states)
    
    while True:
        delta = 0
        
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue
            
            v = V[s]
            new_v = 0
            
            # Sum over all actions weighted by policy
            for action in env.get_actions(s):
                action_prob = policy.get((s, action), 0.25)  # default uniform
                
                # Sum over all possible next states
                for s_prime in range(env.n_states):
                    trans_prob = env.get_transition_prob(s, action, s_prime)
                    reward = env.get_reward(s, action, s_prime)
                    new_v += action_prob * trans_prob * (reward + gamma * V[s_prime])
            
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        
        if delta < theta:
            break
    
    return V
