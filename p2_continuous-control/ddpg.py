import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unityagents import UnityEnvironment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# env = UnityEnvironment(file_name='p2_continuous-control/Reacher_Linux/Reacher.x86_64')


class ValueNetwork(nn.Module):

    def __init__(self, state_size):
        super(ValueNetwork,
              self).__init__()  # invoke nn.Module(father class)'s init

        hidden_state_size = [8 * state_size, 8 * state_size]
        self.dense1 = nn.Linear(state_size, hidden_state_size[0])
        self.dense2 = nn.Linear(hidden_state_size[0], hidden_state_size[1])
        self.dense3 = nn.Linear(hidden_state_size[1], 1)

    def forward(self, state):
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        state_value = self.dense3(x)
        return state_value
        pass


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, max_value=1.0, min_value=-1.0):
        super(ActorNetwork,
              self).__init__()  # invoke nn.Module(father class)'s init

        hidden_state_size = [8 * state_size, 8 * state_size]
        self.dense1 = nn.Linear(state_size, hidden_state_size[0])
        self.dense2 = nn.Linear(hidden_state_size[0], hidden_state_size[1])
        self.dense3 = nn.Linear(hidden_state_size[1], action_size)
        self.max_value = max_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def forward(self, state):
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        action_value = self.dense3(x)
        softmax = nn.Softmax(dim=1)
        actions = softmax(action_value)
        # in this case, all actions between -1 and 1
        return actions * self.value_range + self.min_value
        pass

# multi thread
# n-step bootstrapping
# GAE(Generalized Advantage Estimation)