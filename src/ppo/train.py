import network as net
import src.envs.robot_env as robenv
from torch import *
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np

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

def create_agent(hidden_dimensions, dropout):
    #building the rl agent
    INPUT_FEATURES = env_train.observation_space.shape[0] #input features asks the env, "how many different numbers describe the state"
    HIDDEN_DIMENSIONS = hidden_dimensions
    ACTOR_OUTPUT_FEATURES = env_train.action_space.n #how many acitons are there?
    CRITIC_OUTPUT_FEATURES = 1 #how good is this state?
    DROPOUT = dropout
    actor = net.Actor(INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT) #creating actor network from backbone
    critic = net.Critic(INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT) #creating critic network from backbone
    agent = ActorCritic(actor,critic) #combining both networks into one model
    return agent
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
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    state = env.reset()
    agent.train()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        states.append(state)
        #current state becomes a pytorch tensor and is added to states list
        #then the agent predicts and action and value for current state
        action_pred, value_pred = agent(state)
        #action probabilities are created and then a distribution is created
        #an action is sampled from this distribution and its log probability is calculated
        action_prob =  f.softmax(action_pred, dim = -1)
        dist= distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        #env takes a new step with the sampled action, and relevant values are updated
        state,reward, done, _ = env.step(action.item())
        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
        
    #relevant information is concatenated into tensors
    state = torch.cat(states)
    actions = torch.cat(actions)
    actions_log_probability = torch.cat(actions_log_probability)
    values=  torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, actions_log_probability, advantages, returns


def update_policy(agent,states,actions,actions_log_probability_old,advantages,returns, optimizer,ppo_steps,epsilon,entropy_coefficient):
    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()
    training_results_dataset = TensorDataset(
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns)
    batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False)
    for _ in range(ppo_steps):
        for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
            # get new log prob of actions for all input states
            action_pred, value_pred = agent(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = f.softmax(action_pred, dim=-1)
            probability_distribution_new = distributions.Categorical(
                    action_prob)
            entropy = probability_distribution_new.entropy()
            # estimate new log probabilities using old actions
            actions_log_probability_new = probability_distribution_new.log_prob(actions)
            surrogate_loss = calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    epsilon,
                    advantages)
            policy_loss, value_loss = calculate_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred)
            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def evaluate(env, agent):
        agent.eval()
        reward = []
        done = False
        episode_reward = 0
        state = env.reset()
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred, _ = agent(state)
                action_prob = f.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _ = env.step(action.item())
            episode_reward += reward
        return episode_reward

def run_ppo():
    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []
    agent = create_agent(HIDDEN_DIMENSIONS, DROPOUT)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    for episode in range(1, MAX_EPISODES+1):
        train_reward, states, actions, actions_log_probability, advantages, returns = foward_pass(
                env_train,
                agent,
                optimizer,
                DISCOUNT_FACTOR)
        policy_loss, value_loss = update_policy(
                agent,
                states,
                actions,
                actions_log_probability,
                advantages,
                returns,
                optimizer,
                PPO_STEPS,
                EPSILON,
                ENTROPY_COEFFICIENT)
        test_reward = evaluate(env_test, agent)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))
        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | \
                  Mean Train Rewards: {mean_train_rewards:3.1f} \
                  | Mean Test Rewards: {mean_test_rewards:3.1f} \
                  | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                  | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break    

def plot_losses(policy_losses, value_losses):
    pass

def plot_test_rewards(test_rewards, reward_threshold):
    pass

def plot_train_rewards(train_rewards, reward_threshold):
    pass