from .base_env import BaseEnv
import numpy as np 
from scipy.stats import poisson

class CarRental(BaseEnv):
    """Jake Rental Problem from Sutton & Barto Example 4.2"""

    def __init__(self, max_cars=20, max_moves=5, rent_lambdas=(3,4), return_lambdas=(3,2),move_cost=2,rent_reward=10):
        self.max_cars_per_loc = max_cars
        self.max_car_moves = max_moves
        self.rent_lamdas = rent_lambdas
        self.return_lamdas = return_lambdas
        self.move_cost = move_cost
        self.rent_reward = rent_reward
        self.action = np.arrange(-self.max_car_moves, self.max_car_moves + 1)
        self.states = [(i,j) for i in range(self.max_cars_per_loc + 1) for j in range(self.max_cars_per_loc + 1)]    
    
    def reset(self):
       pass 

    def get_actions(self):
       return [-5,-4,-3,-2,-1,0,1,2,3,4,5]

    
    def get_state(self,index):
        return self.states[index]
        
    def state_valid(self, state):
        if(state[0] < 0 or state[0]>self.max_cars_per_loc or state[1] < 0 or state[1]>self.max_cars_per_loc):
            return False
        return True

    def get_next_state(self,state,action):
        if(action < 0):
            return (state[0]+action,state[1]-action)
        return (state[0]-action,state[1]+action)
        
    def step(self, state, action):
        next_state = self.get_next_state(state,action)
        if(not self.state_valid(next_state)):
            return state,0,False,{}

        reward = self.get_reward(state,action,next_state)
        return next_state, reward, False, {}

    def get_reward(self, state, action,next_state=None):
        #moving cost for that day
        cost = self.move_cost * abs(action)

        probs = {}        
        expected_sales = 0
        cutoff = 11  #to limit poisson calculations
        for d1 in range(cutoff):
            p_d1 = poisson.pmf(d1,self.rent_lamdas[0])
            for d2 in range(cutoff):
                p_d2 = poisson.pmf(d2,self.rent_lamdas[1])
                rented1 = min(state[0]-action,d1)
                rented2 = min(state[1]+action,d2)
                reward = self.rent_reward * (rented1 + rented2)
                expected_sales += p_d1 * p_d2 * reward

                #get return probabilities
                for r1 in range(cutoff):
                    p_r1 = poisson.pmf(r1,self.return_lambdas[0])
                    for r2 in range(cutoff):
                        p_r2 = poisson.pmf(r2,self.return_lambdas[1])
                        new_cars1 = min(next_state[0] + r1,self.max_cars_per_loc)
                        new_cars2 = min(next_state[1] + r2,self.max_cars_per_loc)
                        probs[(new_cars1,new_cars2)] = probs.get((new_cars1,new_cars2),0) + p_d1 * p_d2 * p_r1 * p_r2

        return  expected_sales - cost, probs

    def get_transition_probs(self,state,action,next_state):
        s = self.get_state(state)
        s_prime = self.get_state(next_state)
        if(self.get_next_state(s,action) == s_prime):
            return 1
        return 0

    @property
    def n_actions(self):
        return self.max_car_moves * 2 + 1 

    @property 
    def n_states(self):
       return (self.max_cars_per_loc+1)**2


