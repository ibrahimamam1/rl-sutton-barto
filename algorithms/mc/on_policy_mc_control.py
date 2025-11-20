from algorithms.mc.on_policy_mc_prediction import on_policy_mc_state_action_value_policy_evaluation_
from environments.blackjack import BlackJackEnv
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def on_policy_mc_control(env, n_iterations, epsilon):
    print('Initialising random policy')
    states = env.get_states()
    policy = defaultdict(lambda: np.zeros(env.n_actions))

    # for all possible states
    for s in states:
        policy[s] = np.ones(env.n_actions)/env.n_actions

    print('Learning optimal policy...')
    for i in tqdm(range(n_iterations)):
        #Evaluate current policy
        Q = on_policy_mc_state_action_value_policy_evaluation_(env, policy, discount_factor=1, epsilon=epsilon, n_episodes=5000) 

        #Greedily improve on policy
        for s in states:
            best_action = np.argmax(Q[s])
            for a in range(env.n_actions):
                if a == best_action:
                    policy[s][a] = 1
                else:
                    policy[s][a] = 0 
    return policy

