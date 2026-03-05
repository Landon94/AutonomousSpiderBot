import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim

from src.ppo import network as net
from src.ppo.network import ActorCritic
from src.envs.agent import calculate_returns, calculate_advantages, update_policy


def create_agent(env, hidden_dimensions=None, dropout=None):
    """Create an actor‑critic agent appropriate for the given environment.

    The network implementation in ``src/ppo/network.py`` already defines a fixed
    architecture; the only necessary information is the state and action sizes. The
    optional ``hidden_dimensions`` and ``dropout`` parameters are ignored but kept
    for API compatibility with older versions of the code.
    """
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
        input_features = sample.shape[0]
        action_dim = len(env.joint_indicies)

    actor = net.Actor(input_features, action_dim)
    critic = net.Critic(input_features)
    return net.ActorCritic(actor, critic)

def _unwrap_obs(env_output):
    """Return the raw observation from a gymnasium or custom step/reset output."""
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


def save_agent(agent: net.ActorCritic, path: str):
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

    try:
        from src.envs import spider_env
    except ImportError:
        raise RuntimeError("spider_env is required for training but could not be imported")

    # instantiate environments when training begins
    env_train = spider_env.SpiderEnv(render_mode=None)
    env_test = spider_env.SpiderEnv(render_mode=None)

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