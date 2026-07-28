from environments.base_env import ModelFreeEnv

class RandomWalkEnv(ModelFreeEnv):
    def __init__(self):
        self.state = 3
    @property
    def n_states(self):
        return 7
    
    @property
    def n_actions(self):
        return 2

    def reset(self):
        self.state = 3
        return self.state

    def step(self, action):
        if action == 0 and self.state > 1: #left
            self.state -= 1
            return self.state, 0, False
        
        elif action == 1 and self.state < 5: 
            self.state += 1
            return self.state, 0, False

        elif self.state == 1 and action == 0:
            self.state -= 1
            return self.state, 0, True

        elif self.state == 5 and action == 1:
            self.state += 1
            return self.state, 1, True


