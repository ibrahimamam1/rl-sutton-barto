from environments.base_env import ModelFreeEnv
import numpy as np

class BlackJackEnv(ModelFreeEnv):
    def __init__(self, use_ace = False):
        
        self.player_hand = 0 
        self.dealer_hand = 0 
        self.deck = None 

    @property
    def n_states(self):
        return 21 * 10 # player can range from 2-22, dealer can range from 2-11 
    
    @property
    def n_actions(self):
        return 2 

    def _create_deck(self):
        ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        deck = ranks * 4          # 52 cards
        np.random.shuffle(deck)   # shuffles in-place
        return deck

    def reset(self):
        self.player_hand = 0
        self.dealer_hand = 0

        self.deck = self._create_deck()

        #deal two card to palyer
        self.player_hand += self.deck.pop()
        self.player_hand += self.deck.pop()
        
        #deal one card to dealer
        self.dealer_hand += self.deck.pop()
        return (self.player_hand, self.dealer_hand)

    
    def step(self, action):
        if(action == 1): #hit
            print('Action: Hit')
            self.player_hand += self.deck.pop()
            
            if self.player_hand > 21:
                return (self.player_hand, self.dealer_hand), -1, True #reward=-1 and terminated is True
            else: 
                return (self.player_hand, self.dealer_hand), 0, False
            
        else: #stand
            while(self.dealer_hand < 17):
                self.dealer_hand += self.deck.pop()

            if(self.dealer_hand > 21 or self.dealer_hand < self.player_hand): #player wins
                return (self.player_hand, self.dealer_hand), 1, True 
            elif(self.player_hand == self.dealer_hand): #draw
                return (self.player_hand, self.dealer_hand), 0, True 
            else: #dealer wins 
                return (self.player_hand, self.dealer_hand), -1, True 
