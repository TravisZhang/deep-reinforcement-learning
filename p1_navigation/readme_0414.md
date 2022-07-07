# Project 1: Navigation
### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Environment Setup

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in any folder, and unzip (or decompress) the file. 

3. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

4. Follow the instructions in [this repository](https://github.com/openai/mujoco-py#install-mujoco) to install mujoco, make sure to install version 1.5.0

5. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

6. Clone the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning), and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
    - If you encounter with torch installation error, install version 0.4.0 with following command first:
        ```bash
        pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
        ```
    - If you encounter with tensorflow installation error, install version 1.7.1 with following command first:
        ```bash
        pip install tensorflow==1.7.1
        ```

7. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

8. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

9. Install gym environment
    - Run those lines before installation
    ```bash
    sudo apt-get update --fix-missing
    sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
    sudo apt-get update -y
    sudo apt-get install -y patchelf
    ```
    - Download mujoco from [this link](https://www.roboti.us/download/mjpro150_linux.zip)
    - Copy mujoco content to ~/.mujoco
    - Copy mujoco license mjkey.txt to mujoco path ~/.mujoco
    - Add following line to .bashrc
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/travisz/.mujoco/mjpro150/bin
    ```
    - install gym with following line
    ```bash
    pip3 install gym[all]
    ```

10. (Optional) After all those steps, you can reinstall torch with latest version to access latest features & functionalities like Flatten()

### Training

- You can simply run all cells of the notebook to begin training. There are three cells that are different modes of training:
    - DQN
    - Double DQN
    - Double DQN + Prioritized Experience replay
- The results of each training mode is indicated below each cell, with score printed every 100 episodes and plot of average score vs score per episode
- The detailed report is in DQN report.pdf
