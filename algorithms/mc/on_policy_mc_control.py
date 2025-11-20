from algorithms.mc.mc_prediction import mc_policy_evaluation
from environments.blackjack import BlackJackEnv
import numpy as np
from tqdm import tqdm

def on_policy_mc_control(env, n_iterations):
    print('Initialising random policy')
    states = env.get_states()
    policy = {}

    # for all possible states
    for s in states:
        for i in range(env.n_actions):
            policy[s][i] = 1/env.n_actions

    for i in tqdm(range(n_iterations)):
        #Evaluate current policy
        V = mc_policy_evaluation(env, policy, discount_factor=1, n_episodes=5000) 

        #Greedily improve on policy
                  

