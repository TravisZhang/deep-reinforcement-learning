from signal import pause
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
SamplingPriorityBuffer = 0.1 # extra const buffer for sampling probability
SamplingPriorityOrder = 0.5 # [0,1] 0 for uniform sampling, 1 for complete prioritized sampling
SamplingWeightOrderIncreaseSpeed = 1.0 / 1500.0
SamplingPriorityOrderIncreaseSpeed = 1.0 / 1500.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.sampling_weight_order = 0.001
        self.sampling_priority_order = 0.001
        
        self.prioritized_exp_replay_enabled = False
        self.double_dqn_enabled = False

    def set_prioritized_exp_replay_enabled(self, v):
        self.prioritized_exp_replay_enabled = v
        
    def set_double_dqn_enabled(self, v):
        self.double_dqn_enabled = v

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, SamplingPriorityBuffer)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # print('t_step:', self.t_step)
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if not self.prioritized_exp_replay_enabled:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                else:
                    experiences = self.memory.prioritized_sampling(self.sampling_priority_order)
                    self.prioritized_learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()  # set model to evaluation mode
        # no_grad(): https://blog.csdn.net/sazass/article/details/116668755
        # disable grad calculation for action_values because it's not part of the model
        # if no disabled, action_values's grad will also be calculated during
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # set model to train mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        loss_fn = torch.nn.MSELoss()

        # q learning update
        # !!! first get max q values of next states from q(target)
        if not self.double_dqn_enabled:
            # https://cloud.tencent.com/developer/article/1659274
            # detach is to stop back probagation to reach it, so its requires_grad is always False
            ns_q = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            # !!! double dqn uses different set of params to choose best action and evaluate its value
            # i.e. q(local) for select best action, q(target) for evaluate action value
            # the reason is q(local) is exploring more states(getting more information),
            # so it will less tend to overestimate one state over another
            best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            action_values = self.qnetwork_target(next_states).detach()
            ns_q = action_values.gather(1, best_actions)
        # max(1) returns max vals of dim=1 i.e. torch.return_types.max(values=tensor([...]),indices=tensor([...]))
        # max(1)[0] is the max val tensor
        # unsqueeze(1) is to add a dim to dim index=1 because the first dim must be batch dim
        # !!! then get new q values of current states
        new_q = rewards + (gamma * ns_q * (1 - dones))
        # torch gather: https://blog.csdn.net/Lucky_Rocks/article/details/79676095
        # it means to gather values from dim = 1 at indices = actions
        # dim 0 is the state batch index, dim 1 is the action index
        current_q = self.qnetwork_local(states).gather(1, actions)
        # print('cur shape:', current_q.shape, 'new shape', new_q.shape)
        # !!! q learning update
        loss = loss_fn(current_q, new_q)
        # loss = F.mse_loss(new_q, current_q)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def prioritized_learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones, probs = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        loss_fn = torch.nn.MSELoss()

        # q learning update
        # !!! first get max q values of next states from q(target)
        if not self.double_dqn_enabled:
            # https://cloud.tencent.com/developer/article/1659274
            # detach is to stop back probagation to reach it, so its requires_grad is always False
            ns_q = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            # !!! double dqn uses different set of params to choose best action and evaluate its value
            # i.e. q(local) for select best action, q(target) for evaluate action value
            # the reason is q(local) is exploring more states(getting more information),
            # so it will less tend to overestimate one state over another
            best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            action_values = self.qnetwork_target(next_states).detach()
            ns_q = action_values.gather(1, best_actions)
        # max(1) returns max vals of dim=1 i.e. torch.return_types.max(values=tensor([...]),indices=tensor([...]))
        # max(1)[0] is the max val tensor
        # unsqueeze(1) is to add a dim to dim index=1 because the first dim must be batch dim
        # !!! then get new q values of current states
        new_q = rewards + (gamma * ns_q * (1 - dones))
        # torch gather: https://blog.csdn.net/Lucky_Rocks/article/details/79676095
        # it means to gather values from dim = 1 at indices = actions
        # dim 0 is the state batch index, dim 1 is the action index
        current_q = self.qnetwork_local(states).gather(1, actions)
        # print('cur shape:', current_q.shape, 'new shape', new_q.shape)
        # !!! add weights to q values
        weights = self.add_weights_to_q(current_q, new_q, probs)
        # !!! q learning update
        loss = loss_fn(current_q, new_q)
        # loss = F.mse_loss(new_q, current_q)
        # self.remove_weights(current_q, weights)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        self.update_sampling_weight_order()
        
        self.memory.update_priority(new_q, current_q.detach())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_sampling_weight_order(self):
        # as training continue, the order increases from 0 to 1 as q values converge
        self.sampling_weight_order += SamplingWeightOrderIncreaseSpeed
        self.sampling_weight_order = min(self.sampling_weight_order, 1.0)
        self.sampling_priority_order += SamplingPriorityOrderIncreaseSpeed
        self.sampling_priority_order = min(self.sampling_priority_order, 1.0)

    def add_weights_to_q(self, current_q, new_q, probs):
        N = len(probs)
        weights = 1/(N*probs)**self.sampling_weight_order
        # normalize to increase stability so that they only scale the update downwards
        # print('weights:',weights)
        # print('probs:',probs)
        max_w = torch.clone(max(weights.squeeze())).tolist()
        weights = weights/max_w
        # print('max_w:',max_w)

        current_q *= weights
        new_q *= weights
        return weights
        
    def remove_weights(self, current_q, weights):
        current_q /= weights

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.sampling_indices = None
        self.priorities = np.zeros(buffer_size)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def prioritized_sampling(self, sampling_priority_order):
        memory_len = len(self.memory)
        if self.sampling_indices is None:
            self.priorities[0:memory_len] = np.array(
                [m.priority**sampling_priority_order for m in self.memory])
        else:
            for i in self.sampling_indices:
                self.priorities[i] = self.memory[i].priority**sampling_priority_order
        
        # print('priorities len:', len(self.priorities))
        one_over_sum_of_priorities = 1.0 / sum(self.priorities)
        # print('one_over_sum_of_priorities:', one_over_sum_of_priorities)
        sampling_probs = np.array(
            [p * one_over_sum_of_priorities for p in self.priorities])
        sampling_indices = np.linspace(0, memory_len - 1,
                                       memory_len).astype(int)
        # print('sampling_indices:',len(sampling_indices),'sampling_probs:',len(sampling_probs))
        actual_sampling_indices = [
            np.random.choice(sampling_indices, p=sampling_probs[0:memory_len])
            for i in range(self.batch_size)
        ]
        experiences = [self.memory[i] for i in actual_sampling_indices]
        self.sampling_indices = actual_sampling_indices

        # note that default variables are in ('cpu'), so we need to convert them to device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        sampling_probs = torch.from_numpy(np.vstack([sampling_probs[i] for i in actual_sampling_indices])).to(device)

        return (states, actions, rewards, next_states, dones, sampling_probs)

    def update_priority(self, new_q, current_q):
        if self.sampling_indices is None:
            return
        td_error = new_q - current_q
        priorities = abs(td_error.squeeze()) + SamplingPriorityBuffer
        # print('priorities:',priorities)
        for p, i in zip(priorities, self.sampling_indices):
            # print('p:',p.tolist(),'pri:',self.memory[i].priority)
            self.memory[i] = self.memory[i]._replace(priority=p.tolist())
            # print('p:',p.tolist(),'pri:',self.memory[i].priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



if __name__ == "__main__":
    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
    memory = deque(maxlen=10)
    memory.append(experience([1],[2],[3],[4],[5],[6]))
    print('memory:',memory[0])
    exps = np.array([memory[0]])
    exps = [memory[0]]
    print('state:',exps[0].state)
    exps[0] = exps[0]._replace(state=10)
    print('state:',exps[0].state)
