import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unityagents import UnityEnvironment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# env = UnityEnvironment(file_name='p2_continuous-control/Reacher_Linux/Reacher.x86_64')


def Softmax(x: torch.Tensor):
    exp_x = torch.exp(x)
    sum = torch.sum(exp_x, dim=1)
    return exp_x / sum.view(-1, 1)


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


class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(PolicyNetwork,
              self).__init__()  # invoke nn.Module(father class)'s init

        hidden_state_size = [8 * state_size, 8 * state_size]
        self.dense1 = nn.Linear(state_size, hidden_state_size[0])
        self.dense2 = nn.Linear(hidden_state_size[0], hidden_state_size[1])
        self.dense3 = nn.Linear(hidden_state_size[1], action_size)

    def forward(self, state):
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        action_value = self.dense3(x)
        softmax = nn.Softmax(dim=1)
        probs = softmax(action_value)
        return probs
        pass


class A2C():

    def __init__(self, env: UnityEnvironment):
        self.state_size, self.action_size = ProcessEnvironmentInfo(env)

        self.value_network_local = ValueNetwork(self.state_size).to(device)
        self.policy_network_local = PolicyNetwork(self.state_size,
                                                  self.action_size).to(device)

        self.value_network_target = ValueNetwork(self.state_size).to(device)
        self.policy_network_target = PolicyNetwork(self.state_size,
                                                   self.action_size).to(device)

    def Learn(self, num_episodes, gamma, eps):

        pass

    def EpsilonGreedy(self, probs, eps):
        batch_size = probs.shape[0]
        actions = np.zeros(batch_size)
        for i in range(batch_size):
            # Epsilon-greedy action selection
            if np.random.random() > eps:
                actions[i] = np.argmax(probs.cpu().data.numpy()[0], axis=1)
            else:
                actions[i] = np.random.choice(np.arange(self.action_size))
        return actions

    def DiscreteAct(self, states, eps=None):
        # input: states is a batch of different states
        # output: a batch of best actions by the same size
        states = torch.from_numpy(states).float().to(device)
        probs = self.policy_network_local(states).detach().cpu().data.numpy()
        batch_size = probs.shape[0]

        if eps is not None:
            # Epsilon-greedy action selection
            # Note that we have to convert to numpy, otherwise probs may not sum to 1
            if np.random.random() > eps:
                return np.argmax(probs, axis=1)
            else:
                return np.random.choice(np.arange(self.action_size))
        else:
            # random selection according to policy probs
            action_indices = np.stack(
                [np.arange(self.action_size) for i in range(batch_size)])
            actions = np.stack([
                np.random.choice(action_index, p=prob)
                for action_index, prob in zip(action_indices, probs)
            ])
            return actions
        pass

    def ContinuousAct(self, states, noise_level=1e-3):
        actions = self.policy_network_local(
            states).squeeze().cpu().detach().numpy()
        noises = np.random.randn(actions.shape) * noise_level
        actions += noises
        return actions

    def CollectTrajectories(self,
                            env: UnityEnvironment,
                            policy: PolicyNetwork,
                            num_agents=5,
                            t_max=5,
                            nrand=5,
                            init_states=None):
        # a single env that can have multiple agents working at the same time

        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        brain_name = env.brain_names[0]

        if init_states is None:
            env.reset()
            for i in range(nrand):
                actions = np.random.randn(num_agents, action_size)
                actions = np.clip(actions, -1, 1)
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations  # get next state (for each agent)
                init_states = next_states  # roll over states to next time step

        current_states = init_states
        for i in range(t_max):
            actions = self.ContinuousAct(current_states)
            rewards = env_info.rewards  # get reward (for each agent)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get next state (for each agent)
            current_states = next_states
            
            state_list.append(current_states)
            reward_list.append(rewards)
            action_list.append(actions)
            pass


def ProcessEnvironmentInfo(env: UnityEnvironment):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    print('state size:', state_size, 'action size:', action_size)

    return state_size, action_size


if __name__ == "__main__":
    state_size = 33
    action_size = 4
    batch_size = 10

    # probs = np.random.uniform(0, 1, (8, 4))
    # # softmax = nn.Softmax(dim=1)
    # # probs = softmax(torch.from_numpy(probs))
    # probs = Softmax(torch.from_numpy(probs))
    # print('probs:', probs)

    value_network = ValueNetwork(state_size)
    policy_network = PolicyNetwork(state_size, action_size)

    states = []
    for i in range(batch_size):
        state = np.random.randn(state_size)
        states.append(state)
    states = torch.from_numpy(np.stack(states)).float()
    print('states size:', states.shape)

    values = value_network(states)
    print('values:', values)

    probs = policy_network(states).detach()
    print('probs size:', probs.shape)
    print('probs:', probs)

    prob_sum = torch.sum(probs, dim=1)
    print('prob_sum:', prob_sum)

    for p in probs:
        print('p:', p, 'sum:', torch.sum(p))

    action_indices = np.stack(
        [np.arange(action_size) for i in range(probs.shape[0])])
    print('action_indices:', action_indices)
    actions = np.stack([
        np.random.choice(action_index, p=prob)
        for action_index, prob in zip(action_indices,
                                      probs.cpu().data.numpy())
    ])
    print('actions:', actions)
    actions1 = np.argmax(probs.cpu().data.numpy(), axis=1)
    print('actions1:', actions1)

    pass