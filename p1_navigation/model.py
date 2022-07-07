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

# currently the same as DQN
class QNetworkCNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkCNN, self).__init__() # invoke nn.Module(father class)'s init
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.conv2d1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2d2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d3 = nn.Conv2d(64, 64, 3, stride=1)
        self.flatten = nn.Flatten()
        self.dense4 = nn.Linear(3136, 512)
        self.dense5 = nn.Linear(512, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # print('state shape:',state.shape)
        x = F.relu(self.conv2d1(state.permute(0,3,1,2)))
        x = F.relu(self.conv2d2(x))
        x = F.relu(self.conv2d3(x))
        x = F.relu(self.flatten(x))
        x = F.relu(self.dense4(x))
        action_value = self.dense5(x)
        return action_value
        pass

if __name__ == '__main__':
    # test_model = QNetwork(4,2,0)
    # params = [param for param in test_model.parameters()]
    # param_grads = [p.grad for p in params]
    # print(params)
    # print(param_grads)
    # test_model.zero_grad()
    # print(param_grads)

    state_size = (4,84,84,3)
    action_size = 4
    test_model = QNetworkCNN(state_size, action_size, 0)
    test_image = torch.rand(4,84,84,3)
    action_values = test_model.forward(test_image)
    print('action_values: ',action_values)
