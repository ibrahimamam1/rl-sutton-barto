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


        # Instead of appending, use indexing:
        for action_idx, action in enumerate(env.get_actions(s)):
            q_a = 0
            for s_prime in range(env.n_states):
                trans_prob = env.get_transition_prob(s, action, s_prime)
                reward = env.get_reward(s, action, s_prime)  # singular
                q_a += trans_prob * (reward + gamma * V[s_prime])

            q[action_idx] = q_a  # Assign to index, don't append!

        greedy_action = np.argmax(q)
        new_policy[s][greedy_action] = 1

    return new_policy
