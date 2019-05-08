import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import sys

def plot_rewards(df, smoothing_window=100):
    fig = plt.figure(figsize=(10,5))
    plt.grid(False)
    plt.style.use('seaborn-bright')
    rewards_smoothed = df.rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.gca().legend(df.columns)
    plt.show(fig)

def action_egreedy(Q_state, eps, nA):
  if random.random() < eps:
    action = random.choice(np.arange(nA))
  else:
    action = np.random.choice([action_ for action_, value_ in enumerate(Q_state) if value_ == np.max(Q_state)])
  return action

def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    window = 100 
    avg_rewards = deque(maxlen=num_episodes)
    best_avg_reward = -math.inf
    samp_rewards = deque(maxlen=window)
    rewards = []

    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        initial_state = env.reset()
        action = agent.reset_episode(initial_state, i_episode)
        samp_reward = 0
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            action = agent.step(next_state, reward, done, mode)
            samp_reward += reward
        samp_rewards.append(samp_reward)
        rewards.append(samp_reward)

        # Print episode stats
        if i_episode > 100:
          avg_reward = np.mean(samp_rewards)
          avg_rewards.append(avg_reward)
          if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            print("\rEpisode {}/{} || Best average reward {} ".format(i_episode, num_episodes, best_avg_reward), end="")
            sys.stdout.flush()
        if i_episode == num_episodes: 
            print('\n')
            return rewards    