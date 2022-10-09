from parallelEnv import parallelEnv 
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
# from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import random as rand
import copy
import os, datetime

# https://caw.guaik.io/d/25-jsanimation-attributeerrorhtmlwriter-object-has-no-attribute-temp-names
from IPython.display import HTML

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim.to_jshtml())

RIGHT=4
LEFT=5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
def GetFolderPath(name='result'):
    cur_folder = os.getcwd()
    str_time = datetime.datetime.strftime(datetime.datetime.now(),
                                        '%Y_%m_%d_%H_%M_%S')
    folder_name = '/' + name + '_' + str_time + '/'
    save_path = cur_folder + folder_name
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    return save_path
 
# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    # the L34 to L194(210-16=194) is the middle playground
    # then take the mean(why not sum) of the color channels
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return img

# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        # if we expand at axis=0,then we don't need to swap axes in the end
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)


# function to animate a list of frames
def animate_frames(frames):
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape)==3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)  

    fanim = animation.FuncAnimation(plt.gcf(), \
        lambda x: patch.set_data(frames[x]), frames = len(frames), interval=30)
    
    # display(display_animation(fanim, default_mode='once'))
    display(display_animation(fanim))
    
# play a game and display the animation
# nrand = number of random steps before using the policy
def play(env, policy, time=2000, preprocess=None, nrand=5):
    env.reset()

    # star game
    env.step(1)
    
    # perform nrand random steps in the beginning
    for _ in range(nrand):
        frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT,LEFT]))
        frame2, reward2, is_done, _ = env.step(0)
    
    anim_frames = []
    
    for _ in range(time):
        
        frame_input = preprocess_batch([frame1, frame2])
        prob = policy(frame_input)
        
        # RIGHT = 4, LEFT = 5
        action = RIGHT if rand.random() < prob else LEFT
        frame1, _, is_done, _ = env.step(action)
        frame2, _, is_done, _ = env.step(0)

        if preprocess is None:
            anim_frames.append(frame1)
        else:
            anim_frames.append(preprocess(frame1))

        if is_done:
            break
    
    env.close()
    
    animate_frames(anim_frames)
    return 



# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, tmax=200, nrand=5):
    
    # number of parallel instances
    n=len(envs.ps)
    # print('parallel instances num:',n)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    
    # start all parallel agents
    envs.step([1]*n)
    
    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
        fr2, re2, _, _ = envs.step([0]*n)
    
    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into 
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([fr1,fr2])
        
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()
        
        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)
        
        
        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0]*n)

        reward = re1 + re2
        
        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, \
        action_list, reward_list

# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    # the shape of (stacked) states is (tmax, num_trajs, 2, 80, 80)
    # so the last 3 dims are shape of input of one frame
    states = torch.stack(states)
    # view(-1,2,80,80) combines tmax and num of trajs, =view(tmax*num_trajs,2,80,80)
    # '-1' means to auto determine the number
    # the output shape is (tmax*num_trajs,1)
    # '*' is used to deserialize tuple into seperate numbers
    policy_input = states.view(-1,*states.shape[-3:])
    # =view(tmax, num_trajs)
    return policy(policy_input).view(states.shape[:-3])

# return sum of log-prob divided by T
# same thing as -policy_loss
def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

    ratio = new_probs/old_probs

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(ratio*rewards + beta*entropy)

    
# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)
    
    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=9*9*16
        
        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to 
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        # print('x shape:',x.shape)
        x = F.relu(self.fc1(x))
        # print('x shape:',x.shape)
        return self.sig(self.fc2(x))
    


class TestModel(nn.Module):
    
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=8, stride=4)
        self.ba2d1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.ba2d2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.ba2d3 = nn.BatchNorm2d(64)
        self.size=64*7*7

        self.fc1 = nn.Linear(self.size, int(self.size*0.5))
        self.ba1d1 = nn.BatchNorm1d(int(self.size*0.5))
        self.fc2 = nn.Linear(int(self.size*0.5), int(self.size*0.25))
        self.ba1d2 = nn.BatchNorm1d(int(self.size*0.25))
        self.fc3 = nn.Linear(int(self.size*0.25), 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        
        x = F.relu(self.ba2d1(self.conv1(x)))
        x = F.relu(self.ba2d2(self.conv2(x)))
        x = F.relu(self.ba2d3(self.conv3(x)))
        # flatten the tensor
        x = x.view(-1,self.size)
        print('x shape:',x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x
    
if __name__ == '__main__':
    # test model
    model = TestModel()
    policy = Policy()
    input = []
    tmax = 100
    for i in range(tmax):
        input.append(np.random.randn(80,80,2))
    input = torch.from_numpy(np.asarray(input))
    input = torch.reshape(input, (100,2,80,80)).float()
    print('input shape:',input.shape)
    output=model(input)
    print('output shape:',output.shape)
    
    # test states_to_prob
    traj_list = []
    states_at_t = torch.tensor([np.random.randn(2,80,80) for i in range(15)]).float()
    for i in range(100):
        traj_list.append(copy.deepcopy(states_at_t))
    print('traj list[0].shape:',traj_list[0].shape)
    probs = states_to_prob(policy, traj_list)
    print('probs shape:',probs.shape)
    print('type size:',type(probs.shape))
    print('*shape:',*probs.shape)

    # note that python list(ndarray) is also valid
    a=np.arange(10)
    a_list = [a for i in range(10)]
    print(np.mean(a_list, axis=1))
    
    a = torch.tensor(np.random.randn(10))
    b = torch.tensor(np.random.randn(10))
    c = torch.min(a, b)
    print('a:',a)
    print('b:',b)
    print('c:',c)
    