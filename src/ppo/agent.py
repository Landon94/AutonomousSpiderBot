import torch
from torch import optim, distributions
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from src.ppo import network as net

def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def calculate_surrogate_loss(actions_log_probability_old, actions_log_probability_new, epsilon, advantages):
    advantages = advantages.detach()
    policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()
    surrogate_loss1 = policy_ratio * advantages
    surrogate_loss2 = torch.clamp(policy_ratio, min=1.0 - epsilon, max=1.0 + epsilon) * advantages
    surrogate_loss = torch.min(surrogate_loss1, surrogate_loss2)
    return surrogate_loss

def calculate_losses(surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).sum()
    value_loss = f.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss

def update_policy(agent, states, actions, actions_log_probability_old, advantages, returns,
                  optimizer, ppo_steps, epsilon, entropy_coefficient):
    """Perform several PPO epochs over the collected batch of experience."""
    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0

    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()

    dataset = TensorDataset(states, actions, actions_log_probability_old, advantages, returns)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for _ in range(ppo_steps):
        for batch_states, batch_actions, batch_logp_old, batch_advantages, batch_returns in loader:
            dist, value_pred = agent(batch_states)
            value_pred = value_pred.squeeze(-1)

            batch_entropy = dist.entropy().sum(dim=-1)
            batch_logp_new = dist.log_prob(batch_actions).sum(dim=-1)

            surrogate_loss = calculate_surrogate_loss(
                batch_logp_old,
                batch_logp_new,
                epsilon,
                batch_advantages)

            policy_loss, value_loss = calculate_losses(
                surrogate_loss,
                batch_entropy,
                entropy_coefficient,
                batch_returns,
                value_pred)

            # single backward call to accumulate both gradients correctly
            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

class Agent:
    def __init__(self, n_inputs, n_actions):
        self.actor = net.Actor(n_inputs, n_actions)
        self.critic = net.Critic(n_inputs)
        self.actor_critic = net.ActorCritic(self.actor, self.critic)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)

        # PPO parameters
        self.ppo_steps = 8
        self.epsilon = 0.2
        self.entropy_coefficient = 0.01
        self.discount_factor = 0.99

        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def choose_action(self, obs):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        dist, value = self.actor_critic(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().cpu().numpy(), log_prob.item(), value.item()

    def remember(self, obs, action, log_prob, value, reward, done):
        self.states.append(torch.FloatTensor(obs))
        self.actions.append(torch.FloatTensor(action))
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self):
        if len(self.rewards) == 0:
            return
        # Compute returns and advantages
        returns = calculate_returns(self.rewards, self.discount_factor)
        values = torch.tensor(self.values)
        advantages = calculate_advantages(returns, values)

        # Update policy
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs_old = torch.tensor(self.log_probs)

        update_policy(
            self.actor_critic,
            states,
            actions,
            log_probs_old,
            advantages,
            returns,
            self.optimizer,
            self.ppo_steps,
            self.epsilon,
            self.entropy_coefficient
        )

        # Clear buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def save_models(self):
        torch.save(self.actor_critic.state_dict(), 'agent.pth')

    def load_models(self):
        self.actor_critic.load_state_dict(torch.load('agent.pth'))</content>
