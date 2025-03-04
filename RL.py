import numpy as np
import matplotlib as plt

import simple_rl
from simple_rl.mdp import MDP
from simple_rl.mdp.StateClass import State
from simple_rl.agents import QLearningAgent, RandomAgent, LinearQAgent, DoubleQAgent, AccepterAgent, DQNAgent
from simple_rl.run_experiments import run_agents_on_mdp

def RandomDemandGen():
    d1, d2, d3 = False, False, False
    prob = [0.55, 0.2, 0.15, 0.1]

    demand = np.random.choice([0, 1, 2, 3], p = prob)
    if demand == 1:
        d1 = True
    elif demand == 2:
        d2 = True
    elif demand == 3:
        d3 = True
    return d1, d2, d3

class BookingState(State):
    def __init__(self, c1, c2, t, d1, d2, d3):
        """
        c1: Remaining capacity for leg 1 (Frankfurt to London)
        c2: Remaining capacity for leg 2 (London to New York)
        t: Current time period
        d1: Demand type 1
        d2: Demand type 2
        d3: Demand type 3
        """
        super().__init__(data = (c1, c2, t, d1, d2, d3))

    def is_terminal(self):
        c1, c2, t, d1, d2, d3 = self.data
        return t == 0 or (c1 == 0 and c2 == 0)

    def __hash__(self):
        return hash((self.data))
    
    def __eq__(self, other):
        return self.data == other.data
    
    def __repr__(self):
        c1, c2, t, d1, d2, d3 = self.data
        return f"State(c1={c1}, c2={c2}, t={t}, d1={d1}, d2={d2}, d3={d3})"

def transition_function(state, action):
    c1, c2, t, d1, d2, d3 = state.data

    new_d1, new_d2, new_d3 = RandomDemandGen()

    if t == 0:
        return state
    
    #rejection
    if action == "reject":
        return BookingState(c1, c2, t - 1, new_d1, new_d2, new_d3) #rejecting results in no change in state
    
    #acceptance
    if action == "accept":
        #check capacity
        if d1 == True and c1 > 0: #product 1
            return BookingState(c1 - 1, c2, t - 1, new_d1, new_d2, new_d3)
        elif d2 == True and c2 > 0: #product 2
            return BookingState(c1, c2 - 1, t - 1, new_d1, new_d2, new_d3)
        elif d3 == True and c1 > 0 and c2 > 0: 
            return BookingState(c1 - 1, c2 - 1, t - 1, new_d1, new_d2, new_d3)
        else:
            #invalid action ie rejecting, or auto rejecting requested product 0
            return BookingState(c1, c2, t - 1, new_d1, new_d2, new_d3)

def reward_function(state, action):
    prices = {1: 200, 2: 600, 3: 700}
    c1, c2, t, d1, d2, d3 = state.data

    if action == "accept":
        if d1 == True and c1 > 0:  # Product 1: Frankfurt -> London
            return prices[1]
        elif d2 == True and c2 > 0:  # Product 2: London -> New York
            return prices[2]
        elif d3 == True and c1 > 0 and c2 > 0:  # Product 3: Frankfurt -> New York
            return prices[3]
        
    return 0

class BookingMDP(MDP):
    def __init__(self):
        super().__init__(actions=actions,
                         transition_func=self.booking_transition_func,
                         reward_func=self.booking_reward_func,
                         init_state=initial_state,
                         gamma = 1.0)

    def booking_transition_func(self, state, action):
        next_state = transition_function(state, action)
        return next_state

    def booking_reward_func(self, state, action, next_state = None):
        return reward_function(state, action)

    def __repr__(self):
        return "BookingMDP"  # A simple name to avoid invalid characters

actions = ["accept", "reject"]

initial_d1, initial_d2, initial_d3 = RandomDemandGen()

initial_state = BookingState(c1 = 20, c2 = 40, t = 100, d1 = initial_d1, d2 = initial_d2, d3 = initial_d3)

print(f"Initial state set to: {initial_state}")

ql_agent = QLearningAgent(actions, gamma = 1.0, alpha = 0.1, epsilon = 0.2, anneal = True)
linear_agent = LinearQAgent(actions, num_features = 6, gamma = 1, alpha = 0.01, epsilon = 0.2, anneal = True)
doubleq_agent = DoubleQAgent(actions, gamma = 1, alpha = 0.1, epsilon = 0.2, anneal = True)

accepter_agent = AccepterAgent(actions)
random_agent = RandomAgent(actions)
booking_mdp = BookingMDP()

run_agents_on_mdp([random_agent, accepter_agent, ql_agent, doubleq_agent, linear_agent], booking_mdp, instances = 5, episodes = 5000)