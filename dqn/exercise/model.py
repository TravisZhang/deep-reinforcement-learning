from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://blog.csdn.net/qq_34243930/article/details/107231539
# https://blog.csdn.net/dss_dssssd/article/details/82980222
# https://blog.csdn.net/qq_27825451/article/details/90550890

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__() # invoke nn.Module(father class)'s init
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_state_size = [8*state_size, 8*state_size]
        self.dense1 = nn.Linear(state_size, hidden_state_size[0])
        self.dense2 = nn.Linear(hidden_state_size[0], hidden_state_size[1])
        self.dense3 = nn.Linear(hidden_state_size[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        action_value = self.dense3(x)
        return action_value
        pass


# test_model = QNetwork(4,2,0)
# params = [param for param in test_model.parameters()]
# param_grads = [p.grad for p in params]
# print(params)
# print(param_grads)
# test_model.zero_grad()
# print(param_grads)
