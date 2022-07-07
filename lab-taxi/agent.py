import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.7 # learning rate need to be relatively small
        self.alpha_update_iters = 1000
        self.alpha_adjust_ratio = 0.9
        self.gamma = 1.0 # discount rate btwn [0,1]
        self.i_episode = 1
        self.epsilon_hold_level = 1e-5
        self.epsilon = self.get_epsilon(self.i_episode, self.epsilon_hold_level) # the hold value cannot be too big, otherwise the Q can hardly converge
        self.method = 'expected_sarsa'
        print('method:',self.method,'alpha:',self.alpha,'alpha_update_iters:',self.alpha_update_iters,'alpha_adjust_ratio:',self.alpha_adjust_ratio,'gamma:',self.gamma)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.epsilon_greedy(self.Q, state, self.epsilon)

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
        # self.Q[state][action] += 1
        
        # update Q table
        if self.method == 'expected_sarsa':
            self.Q[state][action] = self.update_Q_expected_sarsa(self.alpha, self.gamma, self.Q, state, action, reward, next_state)
        elif self.method == 'sarsamax':
            next_action = np.argmax(self.Q[next_state])
            self.Q[state][action] = self.update_Q_sarsa(self.alpha, self.gamma, self.Q, state, action, reward, next_state, next_action)
        else:
            next_action = self.epsilon_greedy(self.Q, next_state, self.epsilon)
            self.Q[state][action] = self.update_Q_sarsa(self.alpha, self.gamma, self.Q, state, action, reward, next_state, next_action)
             
        if done:
            self.i_episode += 1
            self.epsilon = self.get_epsilon(self.i_episode, self.epsilon_hold_level)
            self.update_alpha(1e-5)

        
    def select_best_action(self, Q, state):
        return np.argmax(Q[state])

    def get_epsilon(self, i_episode, min_val):
        epsilon = 1 / (i_episode)
        epsilon = max(epsilon, min_val)
        return epsilon
    
    def update_alpha(self, min_val):
        if self.i_episode % self.alpha_update_iters == 0:
            self.alpha *= self.alpha_adjust_ratio
            self.alpha = max(self.alpha, min_val)
            
    def epsilon_greedy(self, Q, state, epsilon):        
        if len(Q.items()) == 0:
            return np.random.choice(np.arange(self.nA))
        
        # if self.i_episode % 1000 == 0:
        #     np.random.seed()
        if np.random.random() < epsilon:
            return np.random.choice(np.arange(self.nA))
        else:
            return self.select_best_action(Q, state)
                    
    def update_Q_sarsa(self, alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step
        Qsa_next = Q[next_state][next_action] if next_state is not None else 0    
        target = reward + (gamma * Qsa_next)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value
        return new_value
    
    def update_Q_expected_sarsa(self, alpha, gamma, Q, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step
        prob_uniform_random = self.epsilon/self.nA
        prob_greedy = 1 - self.epsilon + prob_uniform_random
        probs = np.ones(self.nA) * prob_uniform_random
        probs[self.select_best_action(Q, next_state)] = prob_greedy
        Qsa_next = np.dot(Q[next_state], probs) if (next_state is not None) else 0 
        target = reward + (gamma * Qsa_next)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value
        return new_value