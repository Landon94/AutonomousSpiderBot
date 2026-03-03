from src.ppo import network as net
import torch
from torch import optim, distributions
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np

# environment placeholders; instantiation is deferred to avoid importing heavy dependencies
env_train = None
env_test = None

# actor-critic wrapper around the two network modules
class ActorCritic(nn.Module):
    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state: torch.Tensor):
        # actor returns a distribution object for continuous actions
        dist = self.actor(state)
        value = self.critic(state)
        return dist, value

# we will construct real instances later via `create_agent` instead of using this placeholder
# agent = ActorCritic(net.Actor, net.Critic)

def create_agent(env, hidden_dimensions=None, dropout=None):
    """Create an actor‑critic agent appropriate for the given environment.

    The network implementation in ``src/ppo/network.py`` already defines a fixed
    architecture; the only necessary information is the state and action sizes. The
    optional ``hidden_dimensions`` and ``dropout`` parameters are ignored but kept
    for API compatibility with older versions of the code.
    """
    # support both custom robot_env and gymnasium-style environments
    if hasattr(env, "observation_space"):
        # gymnasium environment
        input_features = env.observation_space.shape[0]
        if hasattr(env.action_space, "n"):
            action_dim = env.action_space.n
        elif hasattr(env.action_space, "shape"):
            action_dim = env.action_space.shape[0]
        else:
            raise ValueError("Unsupported action_space type for create_agent")
    else:
        sample = env.reset()
        # robot_env.reset returns raw state, so sample is the observation
        input_features = sample.shape[0]
        action_dim = len(env.joint_indicies)

    actor = net.Actor(input_features, action_dim)
    critic = net.Critic(input_features)
    return ActorCritic(actor, critic)

def _unwrap_obs(env_output):
    """Return the raw observation from a gymnasium or custom step/reset output."""
    # gymnasium.reset() returns ``(obs, info)``; step may return 4 or 5 elements.
    if isinstance(env_output, tuple) and len(env_output) >= 1:
        return env_output[0]
    return env_output

def _step(env, action):
    """Perform env.step and normalise return values to (obs, reward, done, info)."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info
    elif len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, done, info
    else:
        # some custom envs might just return obs, reward, done
        obs, reward, done = result
        return obs, reward, done, {}

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

def forward_pass(env, agent, discount_factor):
    """Run one episode using the current policy and collect training data.

    Handles both gymnasium and custom environments by unwrapping the observation
    and normalising the ``step`` return signature.
    """
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    raw = env.reset()
    state = _unwrap_obs(raw)
    agent.train()

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        states.append(state_tensor)

        # actor returns a distribution, critic returns value estimate
        dist, value_pred = agent(state_tensor)

        action = dist.sample()  # shape [1, action_dim]
        log_prob_action = dist.log_prob(action).sum(dim=-1)

        # env expects a numpy array/list of floats
        action_np = action.squeeze(0).detach().cpu().numpy()
        state, reward, done, _info = _step(env, action_np)

        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

    # pack collected tensors for training
    states = torch.cat(states)
    actions = torch.cat(actions)
    actions_log_probability = torch.cat(actions_log_probability)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, actions_log_probability, advantages, returns


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

def save_agent(agent: ActorCritic, path: str):
    """Serialize the agent's parameters to disk.

    Only the state_dict is saved so that the file size stays reasonable and a
    compatible architecture can be re-constructed by calling ``create_agent``
    before loading.
    """
    torch.save(agent.state_dict(), path)


def load_agent(path: str, env):
    """Create a new agent matching ``env`` and load parameters from ``path``.

    The saved file is expected to contain a state_dict produced by
    :func:`save_agent`.
    """
    agent = create_agent(env)
    state = torch.load(path)
    agent.load_state_dict(state)
    return agent


def evaluate(env, agent):
    agent.eval()
    done = False
    episode_reward = 0
    raw = env.reset()
    state = _unwrap_obs(raw)
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = agent(state_tensor)
            action = dist.mean  # deterministic policy for evaluation
        action_np = action.squeeze(0).cpu().numpy()
        state, reward, done, _info = _step(env, action_np)
        episode_reward += reward
    return episode_reward

def run_ppo(save_path: str | None = None):
    """Train the PPO agent.

    If ``save_path`` is provided the agent parameters will be written to that
    location once training completes (or immediately upon hitting the reward
    threshold).
    """
    global env_train, env_test

    # import robot_env when necessary to avoid heavy dependencies during testing
    try:
        from src.envs import robot_env as robenv
    except ImportError:
        raise RuntimeError("robot_env is required for training but could not be imported")

    # instantiate environments when training begins
    env_train = robenv.RobotEnv(16, 0)
    env_test = robenv.RobotEnv(16, 0)

    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64  # unused by current network implementation
    DROPOUT = 0.2           # unused as well
    LEARNING_RATE = 0.001

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []

    agent = create_agent(env_train, HIDDEN_DIMENSIONS, DROPOUT)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in range(1, MAX_EPISODES + 1):
        train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(
            env_train,
            agent,
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
            print(f'Episode: {episode:3} | '
                  f'Mean Train Rewards: {mean_train_rewards:3.1f} '
                  f'| Mean Test Rewards: {mean_test_rewards:3.1f} '
                  f'| Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} '
                  f'| Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')

        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            # save model when target is reached if a path has been provided
            if save_path is not None:
                save_agent(agent, save_path)
            break

    # always save final agent if path was given
    if save_path is not None:
        save_agent(agent, save_path)

def plot_losses(policy_losses, value_losses):
    plt.figure()
    plt.plot(policy_losses, label="policy loss")
    plt.plot(value_losses, label="value loss")
    plt.legend()
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()


def plot_test_rewards(test_rewards, reward_threshold):
    plt.figure()
    plt.plot(test_rewards, label="test reward")
    if reward_threshold is not None:
        plt.axhline(reward_threshold, color="r", linestyle="--", label="threshold")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure()
    plt.plot(train_rewards, label="train reward")
    if reward_threshold is not None:
        plt.axhline(reward_threshold, color="r", linestyle="--", label="threshold")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.tight_layout()
    plt.show()