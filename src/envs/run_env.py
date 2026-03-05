import numpy as np
from spider_env import SpiderEnv
import time
from ppo.agent import Agent
import os

log_file = "reward_log2.txt"
reward_history = []


def log_episode_reward(episode, mean_reward):
    with open(log_file, "a") as f:
        f.write(f"Episode {episode}: {mean_reward}\n")


env = SpiderEnv(render_mode="human")

obs, info = env.reset()

input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

agent = Agent(n_inputs=input_dims, n_actions=n_actions)
#agent.load_models()
train = False
episode = 0
episode_reward = 0

for _ in range(1000000):

    action, log_prob, value = agent.choose_action(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward
    done = terminated or truncated

    agent.remember(obs, action, log_prob, value, reward, done)

    if done:
        episode += 1

        reward_history.append(episode_reward)
        mean_reward = np.mean(reward_history)

        log_episode_reward(episode, mean_reward)

        if train:
            agent.learn()

        if episode % 10 == 0 and train:
            print(f"EPISODE: {episode}")
            agent.save_models()

        obs, info = env.reset()
        episode_reward = 0

env.close()