import numpy as np


def policy_improvement(env, policy, V, gamma=1.0):
    """
    improve a policy by greedy action selection with respect to Values

    Returns:
        Pi: Improved Policy
    """
    new_policy = np.zeros((env.n_states, env.n_actions))

    for s in range(env.n_states):
        if s in env.terminal_states:
            continue

        q = np.zeros(env.n_actions)


        
        for action_idx, action in enumerate(env.get_actions(s)):
            next_state, reward, trans_prob, done, info = env.step(s, action)
            q_a = trans_prob * (reward + gamma * V[next_state])

            q[action_idx] = q_a 

        greedy_action = np.argmax(q)
        new_policy[s][greedy_action] = 1

    return new_policy
