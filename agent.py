import numpy as np
from collections import defaultdict, deque
import random
import math
import gym
from utils import action_egreedy

class Agent:
    """Agent that can act on an environment"""

    def __init__(self, env, learning = "sarsa", double = False, train = True, alpha=0.1, gamma=1.0,
                 epsilon_start=1.0, seed=505):

        self.env = env
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.seed = np.random.seed(seed)
        self.learning = learning
        self.double = double
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon_start = epsilon_start  
        
        # Create Q-table
        if self.double:
          self.q_table_1 = np.zeros((self.state_size, self.action_size))
          self.q_table_2 = np.zeros((self.state_size, self.action_size))
        else:
          self.q_table = np.zeros((self.state_size, self.action_size))

    def reset_episode(self, initial_state, step):

        # Gradually decrease exploration rate
        self.epsilon = max(self.epsilon_start / (3*step + 1),0.0001)

        # Decide initial action
        self.last_state = initial_state
        
        if self.double:
          self.last_action = np.argmax(np.mean([self.q_table_1[self.last_state], self.q_table_2[self.last_state]], axis=0))
        else:
          self.last_action = np.argmax(self.q_table[self.last_state])
        
        return self.last_action
    
    def step(self, new_state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""

        if mode == 'test':
            # Test mode: take greedy action
            action = np.argmax(self.q_table[new_state])
            return action
        
        else:
            # Train mode: take a step and return action
            
            # QL step update 
            if self.learning == "q_learning":
              self.q_table[self.last_state, self.last_action] += self.alpha * \
                (reward + self.gamma * max(self.q_table[new_state]) - self.q_table[self.last_state, self.last_action])
              new_action = action_egreedy(self.q_table[self.last_state], self.epsilon, self.action_size)
                          
            # SARSA step update 
            elif self.learning == "sarsa":
              new_action = action_egreedy(self.q_table[new_state], self.epsilon, self.action_size)
              self.q_table[self.last_state, self.last_action] += self.alpha * \
                (reward + self.gamma * self.q_table[new_state, new_action] - self.q_table[self.last_state, self.last_action])
            
            # Expected SARSA step update 
            elif self.learning == "expected_sarsa":
              self.q_table[self.last_state, self.last_action] += self.alpha * \
                (reward + self.gamma * np.mean(self.q_table[new_state]) - self.q_table[self.last_state, self.last_action])
              new_action = action_egreedy(self.q_table[new_state], self.epsilon, self.action_size)
            
            # Double Sarsa step update 
            elif self.learning == "double_sarsa":
              new_action = action_egreedy(np.mean([self.q_table_1[new_state],self.q_table_2[new_state]], axis=0), self.epsilon, self.action_size)
              if random.random() < 0.5:
                self.q_table_1[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * self.q_table_1[new_state, new_action] - self.q_table_1[self.last_state, self.last_action])
              else:
                self.q_table_2[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * self.q_table_2[new_state, new_action] - self.q_table_2[self.last_state, self.last_action])
            
            # Double Expected Sarsa step update 
            elif self.learning == "double_expected_sarsa":
              if random.random() < 0.5:
                self.q_table_1[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * np.mean(self.q_table_2[new_state]) - self.q_table_1[self.last_state, self.last_action])
              else:
                self.q_table_2[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * np.mean(self.q_table_1[new_state]) - self.q_table_2[self.last_state, self.last_action])
              new_action = action_egreedy(np.mean([self.q_table_1[new_state],self.q_table_2[new_state]], axis=0), self.epsilon, self.action_size)
            
            # Double QL step update 
            elif self.learning == "double_q_learning":
              if random.random() < 0.5:
                self.q_table_1[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * self.q_table_2[new_state, np.argmax(self.q_table_1[new_state])] - self.q_table_1[self.last_state, self.last_action])
              else:
                self.q_table_2[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * self.q_table_1[new_state, np.argmax(self.q_table_2[new_state])] - self.q_table_2[self.last_state, self.last_action])
              new_action = action_egreedy(np.mean([self.q_table_1[self.last_state],self.q_table_2[self.last_state]], axis=0), self.epsilon, self.action_size)
            
            else:
              raise ValueError('Learning algorithm not supported')
            
            #rollout state and action
            self.last_state = new_state
            self.last_action = new_action
            return new_action