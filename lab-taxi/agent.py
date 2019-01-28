import numpy as np
from collections import defaultdict
import random
import pdb


class Agent:

    def __init__(self, nA=6, alpha=1, gamma=0.99, eps=0.8):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        ##self.nA = nA
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps 
       
        
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        ####################
        ## epsilon_greedy ##
        #################### 
        if random.random() > self.eps: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(self.nA)
        from monitor import i_episode
        pdb.set_trace()
        self.eps = eps / i_episode 
        
        
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        ##################
        # Expected Sarsa #
        ##################
#         # use epsilon-greedy to select the next action from next state
#         next_action = select_action(next_state)  
        
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        
#         # get value of state, action pair at next time step
#         Qsa_next = np.max(self.Q[next_state])if next_state is not None else 0    
        
        policy_s = np.ones(self.nA) * self.eps / self.nA  # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA) # greedy action
        Qsa_next = np.dot(self.Q[next_state], policy_s)         # get value of state at next time step
        
        target = reward + (self.gamma * Qsa_next)               # construct TD target
        self.Q[state][action] += current + (self.alpha * (target - current))# get update value
