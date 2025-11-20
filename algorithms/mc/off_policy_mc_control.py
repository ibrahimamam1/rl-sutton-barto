from algorithms.mc.off_policy_mc_prediction import off_policy_mc_state_action_value_policy_evaluation_
from environments.blackjack import BlackJackEnv
import numpy as np
from tqdm import tqdm

def off_policy_mc_control(env, n_iterations):
    print('Initialising random policy')
    states = env.get_states()
    target_policy = defaultdict(lambda: np.zeros(env.n_actions))
    behavior_policy = defaultdict(lambda: np.zeros(env.n_actions))

    # for all possible states
    for s in states:
        target_policy[s] = np.ones(env.n_actions)/env.n_actions
        behavior_policy[s] = np.ones(env.n_actions)/env.n_actions

    print('Learning optimal policy...')
    for i in tqdm(range(n_iterations)):
        #Evaluate current policy
        Q = off_policy_mc_state_action_value_policy_evaluation_(env, behavior_policy, discount_factor=1, epsilon=0.2, n_episodes=5000) 

        #Greedily improve on policy
        for s in states:
            best_action = np.argmax(Q[s])
            for a in range(env.n_actions):
                if a == best_action:
                    target_policy[s][a] = 1
                else:
                    target_policy[s][a] = 0 
    print('Policy learned')
    return target_policy

