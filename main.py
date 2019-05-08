import numpy as np
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import run, plot_rewards
from agent import Agent

'''
Solving parameters (best avg reward = 9.75 in approx 62k episodes)
python main.py --learning_algo='sarsa' --plot="True" --episodes=100000 --smoothing_window=100 --alpha=0.01
'''

parser = argparse.ArgumentParser(
    description='Optional arguments for solving a discreete environment. \
                 Example of supported environments are: CliffWalking-v0, Taxi-v2, FrozenLake-v0')

parser.add_argument('--env_name', default = 'Taxi-v2', help='Provide an OpenAI discrete environment name')
parser.add_argument('--learning_algo', default = 'sarsa', help='Provide a learning algorithm')
parser.add_argument('--episodes', default = 10000, help='Provide a number of episodes for training')
parser.add_argument('--alpha', default = 0.1, help='Provide a learning rate')
parser.add_argument('--gamma', default = 1.0, help='Provide a discounting factor')
parser.add_argument('--plot', default = 'False', help='Provide a boolean for plotting results')
parser.add_argument('--smoothing_window', default = 100, help='Provide a plotting smoothing window')
parser.add_argument('--run_all', default = 'False', help='Provide a boolean for run all algorithms and comparing results')
my_namespace = parser.parse_args()

ENV_NAME = my_namespace.env_name
EPISODES_TRAIN = int(my_namespace.episodes)
ALPHA = float(my_namespace.alpha)
GAMMA = my_namespace.gamma
LEARN_ALGO = my_namespace.learning_algo
PLOT = my_namespace.plot
SMOOTHING_WINDOW = int(my_namespace.smoothing_window)
RUN_ALL = my_namespace.run_all
EPS_START = 1.0
SEED = int(505)

if RUN_ALL == 'False':
    if (LEARN_ALGO == 'double_q_learning' 
    or LEARN_ALGO == 'double_sarsa'
    or LEARN_ALGO == 'double_expected_sarsa'):
        double = True
    else:
        double = False

    env = gym.make(ENV_NAME)

    agent = Agent(env, 
                learning = LEARN_ALGO,
                double = double,
                train = True,
                alpha = ALPHA,
                gamma = GAMMA,
                epsilon_start = EPS_START,
                seed = SEED)
    print("Solving for", LEARN_ALGO)
    rewards = run(agent, env, num_episodes = EPISODES_TRAIN)

    if PLOT == "True":
        rewards_df = pd.DataFrame({LEARN_ALGO:rewards})
        plot_rewards(rewards_df, SMOOTHING_WINDOW)

else:
    
    algorithms = {  'q_learning': False,
                    'sarsa': False,
                    'expected_sarsa': False,
                    'double_q_learning': True,
                    'double_sarsa': True,
                    'double_expected_sarsa': True}
    rewards_df = pd.DataFrame()
    for algorithm, flag in algorithms.items():
        
        env = gym.make(ENV_NAME)
        
        agent = Agent(env, 
                learning = algorithm,
                double = flag,
                train = True,
                alpha = ALPHA,
                gamma = GAMMA,
                epsilon_start = EPS_START,
                seed = SEED)
        print("Solving for", algorithm)        
        rewards_df[algorithm] = run(agent, env, num_episodes = EPISODES_TRAIN)
    
    if PLOT == "True":
        plot_rewards(rewards_df, SMOOTHING_WINDOW)