from .base_env import BaseEnv
import numpy as np

class BlackJackEnv(BaseEnv):
    def __init__(self, use_ace = False):
        
        self.player_hand = [] 
        self.dealer_hand = [] 
        
        self.action_space = [0, 1] #0: hit, 1: stand  
        self.deck = None 

    @property
    def n_states(self):
        return 21 * 10 # player can range from 2-22, dealer can range from 2-11 
    
    @property
    def n_actions(self):
        return 2 

    def get_states(self):
        states = []
        for player_sum in range(2, 23):           # Player hand: 2 to 22
            for dealer_showing in range(2, 12):   # Dealer:2  to 11
                states.append((player_sum, dealer_showing))
        return states

 
    def _create_deck(self):
        # A standard 52-card deck. J, Q, K are 10. Ace is 11 (can be adjusted to 1).
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        deck = []
        for suit in suits:
            for rank in ranks:
                if rank in ['J', 'Q', 'K']:
                    deck.append(10)
                elif rank == 'A':
                    deck.append(11) 
                else:
                    deck.append(int(rank))
        
        np.random.shuffle(deck)
        return deck

    def _get_hand_value(self, hand):
        value = sum(hand)
        return value
        
    def _get_obs(self):
        return (self._get_hand_value(self.player_hand), self._get_hand_value(self.dealer_hand))  
    def _get_info(self):
        return {"player total": self._get_hand_value(self.player_hand), "Dealer total": self._get_hand_value(self.dealer_hand)}
    
    def reset(self):
        self.player_hand = []
        self.dealer_hand = []

        self.deck = self._create_deck()

        #deal two card to palyer
        self.player_hand.append(self.deck.pop())
        self.player_hand.append(self.deck.pop())
        
        
        #deal one card to dealer
        self.dealer_hand.append(self.deck.pop()) 

        obs = self._get_obs()
        info = self._get_info()

        print(f'reset env')
        print(f'starting obs = {obs}')
        return obs, info 

    
    def step(self, action):

        terminated = False
        reward = 0

        player_total = self._get_hand_value(self.player_hand)
        dealer_total = self._get_hand_value(self.dealer_hand)

        #update state 
        if(action == 0): #hit
            print('Action: Hit')
            self.player_hand.append(self.deck.pop())
            player_total = self._get_hand_value(self.player_hand)
            
            if player_total > 21:
                terminated = True
                reward = -1 # Player busts
            
        else: #stand

            print('Action: Stand')
            #dealer's turn
            while(self._get_hand_value(self.dealer_hand) < 17):
                self.dealer_hand.append(self.deck.pop())

            dealer_total = self._get_hand_value(self.dealer_hand)
            print(f'obs = ({player_total}, {dealer_total}')
            terminated = True

            if(dealer_total > 21 or dealer_total < player_total): #player wins
                reward = 1
                 
            elif(player_total == dealer_total): #draw
                reward = 0 
            else: #dealer wins 
                reward = -1  

        obs = self._get_obs()
        info = self._get_info() 
        print(f'obs={obs}')
        if(terminated):
            print(f'Episode over. reward = {reward}')
        return obs, reward, terminated, info 
