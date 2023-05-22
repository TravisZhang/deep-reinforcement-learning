import numpy as np
import random
import copy
from collections import namedtuple, deque
from matplotlib import pyplot as plt

from model import Actor, Critic, DeepCritic

import torch
import torch.nn.functional as F
import torch.optim as optim

from unityagents import UnityEnvironment

from replay_buffer import BinaryTree, Experience

import datetime

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 40  # minibatch size
GAMMA = 0.95  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
UPDATE_STEP = 5  # Update every n steps
UPDATE_TIMES = 10  # Update m times per update
AGENT_NUM = 20  # enabled agent number
# Prioritized exp replay
ALPHA = 0.7  # how much prioritization do we apply, 1 for fully applied, 0 for uniform sampling
BETA = 0.8  # how much do we apply IS(Importance-sampling correction), 1 for fully applied, 0 for none

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def TimeStamp():
    return datetime.datetime.now().timestamp()


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 env: UnityEnvironment,
                 save_path=None,
                 random_seed=10000):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.seed = random.seed(random_seed)
        self.state_size, self.action_size = ProcessEnvironmentInfo(env)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size,
                                 random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size,
                                  random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size,
                                   random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size,
                                    random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)

        # Initialize target with local network parameters
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        self.soft_update(self.actor_local, self.actor_target, 1.0)

        # Noise process
        self.noise = OUNoise(self.action_size, random_seed)
        # self.noise = NormalDistributionNoise(self.action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE,
                                   random_seed)

        # Priority replay memory
        self.tree = BinaryTree(BUFFER_SIZE, ALPHA)

        self.t_step = 0

        self.save_path = save_path

        self.print_after_update = False

        self.update_time = 0

    def step(self, state, action, reward, next_state, done, t_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # update_times = min(UPDATE_TIMES, len(self.memory) // BATCH_SIZE)
        # update_times = max(update_times, 1)
        if len(self.memory) > BATCH_SIZE * UPDATE_TIMES and t_step % UPDATE_STEP == 0:
            # print('Updating for ', UPDATE_TIMES, ' times')
            for i in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def priority_step(self, state, action, reward, next_state, done, t_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        experience = Experience(state, action, reward, next_state, done,
                                self.tree.max_priority_alpha)
        self.tree.InsertNode(self.tree.max_priority_alpha, None, experience)

        if self.save_path is not None and t_step < 0:
            input_str = 't_step: ' + str(t_step)
            self.tree.Print(self.save_path, input_str)

        # Learn, if enough samples are available in memory
        if self.tree.Size() > BATCH_SIZE * UPDATE_TIMES and t_step % UPDATE_STEP == 0:
            # print('Updating for ', UPDATE_TIMES, ' times')
            for i in range(UPDATE_TIMES):
                current_time = TimeStamp()
                experiences, weights = self.tree.Sample(BATCH_SIZE, BETA)
                sample_time = TimeStamp() - current_time
                current_time = TimeStamp()
                self.prioritized_learn(experiences, weights, GAMMA)
                learn_time = TimeStamp() - current_time
            input_str = '\rt_step: {:d} \t, sample_time: {:.3f} \tupdate_time: {:.3f} \tlearn_time: {:.3f} \tsum: {:.3f} \tsize: {:d}'.format(
                t_step, sample_time, self.update_time, learn_time,
                self.tree.Sum(), self.tree.Size())
            print(input_str, end="                        ")
            if self.print_after_update:
                self.tree.Print(self.save_path, input_str)
                # self.print_after_update = False
                # self.save_path = None

    # Deprecated
    def multi_agent_step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        agent_index = 0
        for state, action, reward, next_state, done in zip(
                states, actions, rewards, next_states, dones):
            if agent_index < AGENT_NUM:
                self.memory.add(state, action, reward, next_state, done)
                agent_index += 1

        self.t_step += 1
        self.t_step %= UPDATE_STEP

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for i in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def multi_agent_act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        # print('states shape:',states.shape)
        # print('actions shape:',actions.shape)
        if add_noise:
            noises = [self.noise.sample() for action in actions]
            noises = np.asarray(noises)
            actions += noises
            np.clip(actions, self.actor_local.min_value, self.actor_local.max_value)
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next).detach()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def prioritized_learn(self, experiences, weights, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next).detach()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # We take [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) instead of
        # mean squared error loss because it is less sensitive to outliers
        huber_loss = torch.nn.SmoothL1Loss()
        critic_losses = huber_loss(Q_expected, Q_targets)
        critic_loss = torch.mean(weights * critic_losses)
        # Compute TD-error
        td_error = Q_targets - Q_expected.detach()
        priorities = np.abs(td_error.cpu().numpy()) + 1e-6
        priorities = np.squeeze(priorities)
        # print('priority shape: ', priorities.shape)
        # print(priorities)
        # Update priorities to samples
        current_time = TimeStamp()
        self.tree.UpdatePriorities(priorities, self.save_path)
        self.update_time = TimeStamp() - current_time
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu -
                           x) + self.sigma * (np.random.rand(*x.shape) - 0.5)
        self.state = x + dx
        return self.state


class NormalDistributionNoise:

    def __init__(self, size, seed, mu=0.0, sigma=0.05):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.seed = random.seed(seed)

    def sample(self) -> np.array:
        """Update internal state and return it as a noise sample."""
        return self.sigma * np.random.randn(self.size) + self.mu


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences
                       if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences
                       if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def ProcessEnvironmentInfo(env: UnityEnvironment):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # get action & state size
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    print('state size:', state_size, 'action size:', action_size)

    return state_size, action_size


def plot_result(scores, scores_avg, actual_target_score):
    target_score_curve = np.ones(len(scores)) * actual_target_score
    fig = plt.figure(figsize=[15, 10])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plt.xlabel("episode"), plt.ylabel("score")
    ax.plot(scores)
    ax.plot(scores_avg)
    ax.plot(target_score_curve)
    ax.legend(['score', 'avg_score', 'target_score'])
    plt.show()


if __name__ == "__main__":
    state_size = 24
    action_size = 2
    batch_size = 10
    num_agents = 2

    actor = Actor(state_size, action_size, 10000, 2, -2)
    critic = Critic(state_size, action_size, 10000)

    states = []
    for i in range(num_agents):
        state = np.random.randn(state_size)
        states.append(state)
    states = torch.from_numpy(np.stack(states)).float()
    print('states size:', states.shape)

    actions = actor(states)
    print('actions: ', actions)

    env = UnityEnvironment(file_name='p3_collab-compet/Tennis_Linux_NoVis/Tennis.x86_64')

    agent = Agent(env)

    actions = agent.multi_agent_act(states.numpy())

    print('multi_agent_act actions:\n', actions)
    print('actions shape:', actions.shape)

    actions = np.random.randn(num_agents,
                              action_size)  # select an action (for each agent)
    actions = np.clip(actions, -1, 1)  # all actions between -1 and 1

    print('random actions:\n', actions)
    print('actions shape:', actions.shape)

    scores = np.ones(10)
    scores_avg = np.ones(10) * 2
    actual_target_score = 3

    plot_result(scores, scores_avg, actual_target_score)

    normal_noise = NormalDistributionNoise(5, 10000)
    noise_sample = np.array([normal_noise.sample() for i in range(10)])
    print('normal noise: ', noise_sample)

    ou_noise = OUNoise(5, 10000)
    noise_sample = np.array([ou_noise.sample() for i in range(10)])
    print('ou noise: ', noise_sample)
