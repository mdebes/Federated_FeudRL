# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:31:22 2025

@author: mdbs1
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from crazy_rl.multi_agent.numpy.catch.catch import Catch
import pygame
import time
from crazy_rl.multi_agent.numpy.base_parallel_env import CLOSENESS_THRESHOLD

# -----------------------------
# Define a simple policy network.
# -----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        # Initialize log_std as a learnable parameter (one per action dimension)
        self.log_std = nn.Parameter(torch.zeros(action_size))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mean = torch.tanh(self.fc2(x))  # mean in [-1, 1]
        std = torch.exp(self.log_std)   # standard deviation, always positive
        return mean, std


# -----------------------------
# Federated agent that holds the policy and its optimizer.
# -----------------------------
class FederatedAgent:
    def __init__(self, state_size, action_size, lr=5e-4):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

# -----------------------------
# Helper: Compute discounted returns.
# -----------------------------
def compute_returns(rewards, gamma=0.995):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns


# -------------------------------
# Randomize environment parameters.
# -------------------------------    

def random_target_speed(min_speed=0.1, max_speed=0.5):
    """
    Generates a random target speed between min_speed and max_speed.
    
    :param min_speed: Minimum target speed (default 0.1).
    :param max_speed: Maximum target speed (default 0.3).
    :return: A random float between min_speed and max_speed.
    """
    return np.random.uniform(min_speed, max_speed)

def random_env_size(min_size, max_size):
    """
    Generates a random environment size as an integer.
    This assumes the environment is square, so one integer represents both width and height.
    
    :param min_size: Minimum size.
    :param max_size: Maximum size.
    :return: A random integer between min_size and max_size.
    """
    return np.random.randint(min_size, max_size + 1)

def random_init_flying_pos_with_min_separation(num_drones, xy_range, z_range, min_sep):
    """
    Generates random initial flying positions for each drone such that each new position 
    is at least `min_sep` (in the xy–plane) away from all previously sampled positions.
    
    :param num_drones: Number of drones.
    :param xy_range: Tuple (min, max) for the x- and y–coordinates.
    :param z_range: Tuple (min, max) for the z–coordinate.
    :param min_sep: Minimum separation distance in the xy–plane between any two drones.
    :return: A numpy array of shape (num_drones, 3).
    """
    positions = []
    attempts = 0
    max_attempts = 1000  # To avoid infinite loops
    while len(positions) < num_drones:
        candidate = np.array([
            np.random.uniform(xy_range[0], xy_range[1]),
            np.random.uniform(xy_range[0], xy_range[1]),
            np.random.uniform(z_range[0], z_range[1])
        ])
        # Check that candidate is at least min_sep away from every accepted position (using only x,y)
        if all(np.linalg.norm(candidate[:2] - pos[:2]) >= min_sep for pos in positions):
            positions.append(candidate)
        attempts += 1
        if attempts > max_attempts:
            raise ValueError("Unable to sample sufficient positions with the specified minimum separation.")
    return np.array(positions)

def random_init_target_location_with_min_separation(xy_range, z_range, drone_positions, min_sep):
    """
    Generates a random initial target location that is at least `min_sep` away (in the xy–plane)
    from all drone positions.
    
    :param xy_range: Tuple (min, max) for the x- and y–coordinates.
    :param z_range: Tuple (min, max) for the z–coordinate.
    :param drone_positions: A numpy array of drone positions (shape (num_drones, 3)).
    :param min_sep: Minimum separation distance in the xy–plane between the target and any drone.
    :return: A numpy array of shape (3,) representing the target location.
    """
    attempts = 0
    max_attempts = 1000
    while True:
        candidate = np.array([
            np.random.uniform(xy_range[0], xy_range[1]),
            np.random.uniform(xy_range[0], xy_range[1]),
            np.random.uniform(z_range[0], z_range[1])
        ])
        if all(np.linalg.norm(candidate[:2] - pos[:2]) >= min_sep for pos in drone_positions):
            return candidate
        attempts += 1
        if attempts > max_attempts:
            raise ValueError("Unable to sample a target location with the specified minimum separation.")

def randomize_catch_parameters(num_drones=2,
                               env_size_range=(2, 5),
                               z_range_flying=(1, 2),
                               z_range_target=(0.2, 3),
                               min_sep=2):
    """
    Randomizes initialization parameters for the Catch environment ensuring that the x and y 
    coordinates for both drones and the target are within the environment boundaries and 
    that they are separated by at least `min_sep` in the xy–plane.
    
    :param num_drones: Number of drones.
    :param env_size_range: Tuple (min, max) for the environment’s square size (env_size is used as the absolute bound).
    :param z_range_flying: Tuple for the z–coordinate range for drone positions.
    :param z_range_target: Tuple for the z–coordinate range for the target location.
    :param min_sep: Minimum separation distance in the xy–plane (now assumed to be 2).
    :return: A dictionary with keys:
             - "init_flying_pos": numpy array of shape (num_drones, 3)
             - "init_target_location": numpy array of shape (3,)
             - "env_size": int
    """
    # First, determine the environment size.
    env_size = random_env_size(env_size_range[0], env_size_range[1])
    
    # For a square environment, the x and y coordinates are within [-env_size, env_size].
    xy_range = (-env_size, env_size)
    
    init_flying_pos = random_init_flying_pos_with_min_separation(num_drones, xy_range, z_range_flying, min_sep)
    init_target_location = random_init_target_location_with_min_separation(xy_range, z_range_target, init_flying_pos, min_sep)
    
    return {
        "init_flying_pos": init_flying_pos,
        "init_target_location": init_target_location,
        "env_size": env_size
    }

# -----------------------------
# Run one episode using a policy-gradient approach.
# agent_dict is a dictionary mapping agent_id (e.g. "agent_0") to a FederatedAgent.
# -----------------------------
def run_episode(agent_dict, env, gamma=0.995):
    # Create dictionaries to collect log probabilities and rewards.
    log_probs_dict = {agent_id: [] for agent_id in agent_dict.keys()}
    rewards_dict = {agent_id: [] for agent_id in agent_dict.keys()}
    entropy_dict  = {agent_id: [] for agent_id in agent_dict.keys()}
    
    # Active mask: True means the agent is still collecting experience.

    obs_dict, _ = env.reset()
    state_dict = obs_dict  # mapping agent_id -> observation
    
    done = False


    while not done:
        actions = {}
        current_log_probs = {}
        current_entropies = {}

        # Loop over active agents. We assume that env.agents matches keys in agent_dict.
        for agent_id in env.agents:
            agent = agent_dict[agent_id]
            state_tensor = torch.FloatTensor(state_dict[agent_id]).unsqueeze(0)
            mean, std = agent.policy(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            actions[agent_id] = action.detach().cpu().numpy()[0]
            current_log_probs[agent_id] = log_prob
            current_entropies[agent_id] = entropy



        # Step the environment
        next_obs_dict, reward_dict, terminated, truncated, _ = env.step(actions)
        
        # Combine terminated and truncated into a done dictionary.
        done_dict = {agent: terminated[agent] or truncated[agent] for agent in env.agents}

        # For active agents, record experience; for inactive ones, do nothing.
        for agent_id in env.agents:
            log_probs_dict[agent_id].append(current_log_probs[agent_id])
            rewards_dict[agent_id].append(reward_dict[agent_id])
            entropy_dict[agent_id].append(current_entropies[agent_id])

        done = all(done_dict.get(agent_id, False) for agent_id in env.agents)
        state_dict = next_obs_dict  # update state

    total_rewards = {agent_id: sum(rewards_dict[agent_id]) for agent_id in log_probs_dict.keys()}
    return log_probs_dict, rewards_dict, total_rewards, entropy_dict

# -----------------------------
# Train for one episode and update the agents.
# -----------------------------
def train_episode(agent_dict, env, entropy_coef=0.01):
    log_probs_dict, rewards_dict, total_rewards, entropy_dict = run_episode(agent_dict, env)
    for agent_id in agent_dict.keys():
        returns = compute_returns(rewards_dict[agent_id])
        loss_terms = []
        for log_prob, R in zip(log_probs_dict[agent_id], returns):
            loss_terms.append(-log_prob * R)
        if loss_terms:
            loss = torch.cat(loss_terms).sum()
            total_entropy = torch.stack(entropy_dict[agent_id]).sum()
            loss = loss - entropy_coef * total_entropy
            agent = agent_dict[agent_id]
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
    return total_rewards

# -----------------------------
# Federated averaging: average the policy network parameters across all agents.
# client_agents_list is a list of agent dictionaries (one per client).
# -----------------------------
def federated_average(client_agents_list):
    # Flatten all agents into one list.
    all_agents = [agent for client_dict in client_agents_list for agent in client_dict.values()]
    num_agents = len(all_agents)
    global_state_dict = {}
    for key in all_agents[0].policy.state_dict().keys():
        global_state_dict[key] = sum(agent.policy.state_dict()[key] for agent in all_agents) / num_agents
    # Update every agent with the averaged (global) state.
    for agent in all_agents:
        agent.policy.load_state_dict(global_state_dict)
        
# -----------------------------
# Calculate entroy decay
# -----------------------------
def entropy_schedule(current_round, total_rounds, initial_coef=0.01, final_coef=0.001):
    # Linear decay schedule:
    return initial_coef - (initial_coef - final_coef) * (current_round / total_rounds)


# -----------------------------
# Main federated training loop.
# -----------------------------
def federated_training(num_clients=4, comm_rounds=50, episodes_per_round=10):
    agents_per_client = 2  # 2 agents per client
    client_agents_list = []  # List of agent dictionaries, one per client.
    client_envs = []         # List of corresponding environments.
    
    # Reset the provided environment to get an initial state sample.
    state_size = 3 * (agents_per_client + 1)
    action_size = 3
    
    # Timer start.
    start_time = time.time()

    # Initialize one environment and a dictionary of agents per client.
    for i in range(num_clients):
        params = randomize_catch_parameters()
        local_env = Catch(
            drone_ids=np.array([0, 1]),
            render_mode=None,
            init_flying_pos=params["init_flying_pos"],
            init_target_location=params["init_target_location"],
            size=params["env_size"],
            target_speed = random_target_speed()
        )
        local_state, _ = local_env.reset()
        local_agents = {}
        # Create 2 agents per client named "agent_0" and "agent_1".
        for j in range(agents_per_client):
            agent_id = f"agent_{j}"
            local_agents[agent_id] = FederatedAgent(state_size, action_size)
        client_agents_list.append(local_agents)
        client_envs.append(local_env)
    
    all_round_rewards = []
    initial_entropy_coef = 0.01
    final_entropy_coef = 0.001
    # Main federated training loop.
    for round in range(comm_rounds):
        round_rewards_by_client = []  # Average reward per client for this round.
        for local_agents, local_env in zip(client_agents_list, client_envs):
            # For each client, collect rewards for each agent over multiple episodes.
            client_episode_rewards = {aid: [] for aid in local_agents.keys()}
            for ep in range(episodes_per_round):
                current_entropy_coef = entropy_schedule(round, comm_rounds, initial_entropy_coef, final_entropy_coef)
                total_rewards = train_episode(local_agents, local_env, current_entropy_coef)
                for aid, r in total_rewards.items():
                    client_episode_rewards[aid].append(r)
            # Compute average reward per client (average over agents).
            client_avg_rewards = [np.mean(client_episode_rewards[aid]) for aid in local_agents.keys()]
            avg_client_reward = np.mean(client_avg_rewards)
            round_rewards_by_client.append(avg_client_reward)
        avg_round_reward = np.mean(round_rewards_by_client)
        all_round_rewards.append(avg_round_reward)
        
        # Running timer.
        running_time = int((time.time() - start_time) / 60)
        
        print(f"\rRound {round+1}/{comm_rounds} - Average Reward: {avg_round_reward:.2f}. Training time: {running_time} minutes.", end="", flush=True)
        federated_average(client_agents_list)
    
    # Close all environments.
    for local_env in client_envs:
        local_env.close()
    
    return all_round_rewards, client_agents_list

def execute_episode(agents, env):
    """
    Runs trained federated agents in the environment and saves the episode as an MP4 video.
    
    :param agents: List of trained federated agents
    :param env: The CrazyRL environment (must support "human" render mode)
    """
    # Initialize environment in "human" mode
    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):  # Ensure correct unpacking
        obs_dict, _ = obs_dict

    done = False

    while not done:
        pygame.time.wait(100)
        env.render()  # Ensure rendering occurs.
        pygame.display.flip()  # Force PyGame to update.

        actions = {}
        # Iterate over active agents.
        for i, agent_id in enumerate(list(env.agents)):
            if agent_id in obs_dict:
                state_tensor = torch.FloatTensor(obs_dict[agent_id]).unsqueeze(0)
                # Select corresponding agent (or cycle through them)
                agent = agents[i % len(agents)]
                mean, std = agent.policy(state_tensor)
                action = mean.detach().cpu().numpy()[0]
                print(f"\rAgent {agent_id} mean: {mean} action: {action}", end="", flush=True)
                actions[agent_id] = action

        # Step the environment
        obs_dict, _, done_dict, _, _ = env.step(actions)
        done = all(done_dict.get(agent_id, False) for agent_id in env.agents)
        
    # Before closing, extract target position from the observations.
    # According to _compute_obs in Catch, each observation is constructed as:
    # [agent_position (3), target_position (3), ...]
    # We assume all agents observe the same target location.
    first_key = list(obs_dict.keys())[0]
    target_pos = obs_dict[first_key][3:6]  # Extract target position from first agent's observation.
    
    # Check if any agent is within CLOSENESS_THRESHOLD of the target or another agent.
    caught = False # Boolean flag for target capture.
    crash = False # Boolean flag for agent collisions.
    
    for agent_id in obs_dict:
        agent_pos = obs_dict[agent_id][:3]  # The first 3 values are the agent's position.
        if np.linalg.norm(agent_pos - target_pos) < CLOSENESS_THRESHOLD:
            caught = True
            break
    
    agent_ids = list(obs_dict.keys())
    
    if not caught:
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                pos1 = obs_dict[agent_ids[i]][:3]
                pos2 = obs_dict[agent_ids[j]][:3]
                if np.linalg.norm(pos1 - pos2) < CLOSENESS_THRESHOLD:
                    crash = True
                    break
            if crash:
                break

    env.close()
    
    return caught, crash

# Test model function

def test_model(client_agents_list, episodes=10):
    
    # For testing, select the agents from the first client:
    trained_agents = list(client_agents_list[0].values())
    
    target_caught = []
    agent_crash = []
    
    for i in range (0,episodes):
        params = randomize_catch_parameters()
        
        test_env = Catch(
            drone_ids=np.array([0, 1]),
            render_mode="human",
            init_flying_pos=params["init_flying_pos"],
            init_target_location=params["init_target_location"],
            size=params["env_size"],
            target_speed=random_target_speed()
        )
        
        # Now execute an episode using the unchanged execute_episode function.
        
        caught, crash = execute_episode(trained_agents, test_env)
        target_caught.append(caught)
        agent_crash.append(crash)
    
    success = sum(target_caught)
    failure = sum(agent_crash)
    success_rate = success / episodes * 100
    failure_rate = failure / episodes * 100
    
    print(f"\rSuccess rate: {success_rate} Crash rate: {failure_rate} Number of episodes: {episodes}.", end="", flush=True)


# -----------------------------
# Entry point: run federated training and plot progress.
# -----------------------------
if __name__ == '__main__':
    # Initialize an environment for sampling.
        
    rewards, client_agents_list = federated_training(num_clients=4, comm_rounds=1000, episodes_per_round=10)
    
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, marker='o')
    plt.xlabel('Communication Round')
    plt.ylabel('Average Episode Reward')
    plt.title('Federated RL Training Progress on CrazyRL (Catch Mode)')
    plt.grid(True)
    plt.show()
    
    test_model(client_agents_list, 10)



