import network as net
import src.envs.robot_env as robenv
from torch import *
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt

env = robenv.RobotEnv(16, 0)

#actor critic class and calculations - will remove once officially implemented

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
    


agent = ActorCritic(net.Actor, net.Critic)

def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward =  r + cumulative_reward * discount_factor
        returns.insert(0,cumulative_reward)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean())/ advantages.std()
    return advantages

def calculate_surrogate_loss(actions_log_probability_old, actions_log_probability_new, epsilon, advantages):
    advantages = advantages.detach()
    policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()
    surrogate_loss1 = policy_ratio * advantages
    surrogate_loss2= torch.clamp(policy_ratio, min=1.0-epsilon, max=1.0+epsilon) * advantages
    surrogate_loss=  torch.min(surrogate_loss1,surrogate_loss2)
    return surrogate_loss


def calculate_losses(surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss=  -(surrogate_loss + entropy_bonus ).sum()
    value_loss=  f.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss

def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    return states, actions, actions_log_probability, values, rewards, done, episode_reward

def foward_pass(env, agent, optimizer, discount_factor):
    pass

def update_policy(agent,states,actions,actions_log_probability_old,advantages,returns, optimizer,ppo_steps,epsilon,entropy_coefficient):
    pass

def evaluate(env, agent):
    pass

def run_ppo():
    pass

def plot_losses(policy_losses, value_losses):
    pass

def plot_test_rewards(test_rewards, reward_threshold):
    pass

def plot_train_rewards(train_rewards, reward_threshold):
    pass