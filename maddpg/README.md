1. To run the code, please use the command "./run_training.sh". The bash script cleans up and DELETE previous runs. The script is necessary because we need an extra command to ensure image rendering is possible remotely. Training takes about two hour. If you run locally on your own computer. Be sure to increase the number of parallel agents to the number of cores your computer have in main.py. GPU does not help that much in the computation.

2. To see a visualization of the results, run the script "./run_tensorboard.sh". A link will appear, and direct your browser to that link to see rewards over time and other statistics

3. The trained models are stored in "model_dir" by default. You can also find .gif animations that show how the agents are performing! The gif file contains a grid of separate parallel agents.

4. To understand the goal of the environment: blue dots are the "good agents", and the Red dot is an "adversary". All of the agents' goals are to go near the green target. The blue agents know which one is green, but the Red agent is color-blind and does not know which target is green/black! The optimal solution is for the red agent to chase one of the blue agent, and for the blue agents to split up and go toward each of the target.

For this Lab, you will train an agent to solve the Physical Deception problem.

This is an ungraded project, Feel free to explore various parameters and see how it affects the way agents approach the problem.

## Goal of the environment
Blue dots are the "good agents", and the Red dot is an "adversary". All of the agents' goals are to go near the green target. The blue agents know which one is green, but the Red agent is color-blind and does not know which target is green/black! The optimal solution is for the red agent to chase one of the blue agent, and for the blue agents to split up and go toward each of the target.

## Running within the workspace ( Recommended Option)
- No explicit setup commands need to run by you, we have taken care of all the installations in this lab, enjoy exploration.
- ./run_training.sh Let's you run the program based on the parameters provided in the main program.
- ./run_tensorboard.sh will give you an URL(should be http://localhost:3000 or http://127.0.0.1:3000) to view the dashboard where you would have visualizations to see how your agents are performing. Use this as a guide to know how the changes you made are affecting the program.
- Folder named Model_dir would store the episode-XXX.gif files which show the visualization on how your agent is performing.
## Running on your own computer
- If you choose to run the program on your computer, you should download the files from the workspace and all the above commands should work the same except for few installations below.
- Use of GPU wouldn't impact the training time for this program, Instead, Multicore environments would be a better choice to increase the training speed.

## Requirements
- sudo apt-get install xvfb
- pip install tensorflow==1.7.1
- [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
- [PyTorch](http://pytorch.org/), version: 0.3.0.post4 (pip install torch==0.4.0)
- [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4 (pip install gym==0.10.5)
- [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging) (pip install tensorboardX)
- pip install pyglet==1.2.4
- If you encounter with tersorboard running error, please [reinstall protobuf](https://blog.csdn.net/liangllhahaha/article/details/104658397) and install grpcio
  - pip uninstall protobuf
  - pip install protobuf==3.5.2
  - pip install grpcio==1.11.0
## To Experiment
- Feel free to clear the model_dir and log folder and start training on your own to see how your agent performs. ./clean.sh should help you accomplish this goal.
- This lab is meant to prepare you for the final project, writing your own functions in maddpg.py will improve your learning curve.
- Also experiment with parameter tuning in main.py, Make note that a larger number of episodes would mean greater training time.
- Lab might take more than one hour to train depending on how the parameters are tuned.