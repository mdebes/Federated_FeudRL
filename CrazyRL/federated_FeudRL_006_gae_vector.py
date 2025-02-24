# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:27:35 2025

@author: mdbs1

Federated FeudRL Implementation

This script implements a federated feudal reinforcement learning approach:
  - A single global ManagerAgent computes goals based on a global state.
  - Multiple clients (here, 4 clients) each have 2 worker agents (sharing one network per client)
    that use the manager’s goal to select actions in a local environment (Catch).
  - Each client runs several local feudal RL episodes.
  - After each communication round, the local worker networks are averaged via federated averaging.
  - In addition, FedProx is applied during the local worker updates to keep client models close to
    the global model (see Li et al., 2020).

References:
    - Feudal RL implementation based on CrazyRL:
      https://github.com/ffelten/CrazyRL/tree/main
    - Federated Averaging:
      McMahan, Brendan, et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data."
      https://arxiv.org/abs/1602.05629
    - FedProx:
      Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020).
      https://arxiv.org/abs/1812.06127
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import time
import copy

# Import the Catch environment and closeness constant.
from crazy_rl.multi_agent.numpy.catch.catch import Catch
from crazy_rl.multi_agent.numpy.base_parallel_env import CLOSENESS_THRESHOLD

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")

# =============================================================================
# Network Definitions (Manager & Worker)
# =============================================================================

class ManagerNetwork(nn.Module):
    def __init__(self, state_size, goal_size, num_agents, xy_scale, z_scale, hidden_size=128):
        super(ManagerNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_mean = nn.Linear(hidden_size, num_agents * goal_size)
        self.fc_std = nn.Linear(hidden_size, num_agents * goal_size)
        self.num_agents = num_agents
        self.goal_size = goal_size
        self.xy_scale = xy_scale    # Environment's size for x and y
        self.z_scale = z_scale      # Fixed scale for z

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        raw_mean = self.fc_mean(x)
        raw_std = self.fc_std(x)
        std = F.softplus(raw_std) + 1e-5
        # Reshape to [batch_size, num_agents, goal_size]
        raw_mean = raw_mean.view(-1, self.num_agents, self.goal_size)
        std = std.view(-1, self.num_agents, self.goal_size)
        # Scale x and y using the environment size and z using the fixed z scale.
        xy_mean = torch.tanh(raw_mean[:, :, :2]) * self.xy_scale
        # Map tanh output from [-1,1] to [0, z_scale]
        z_mean = (torch.tanh(raw_mean[:, :, 2]) + 1) / 2 * self.z_scale
        mean = torch.cat([xy_mean, z_mean.unsqueeze(-1)], dim=-1)
        return mean, std

class WorkerNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(WorkerNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_std = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 1e-5
        return mean, std

# Value networks (Critics) for actor-critic updates.
class ManagerValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ManagerValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_value(x)

class WorkerValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(WorkerValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_value(x)

# =============================================================================
# Agent Definitions
# =============================================================================

class ManagerAgent:
    def __init__(self, state_size, goal_size, num_agents, xy_scale, z_scale, lr=5e-4):
        self.policy_net = ManagerNetwork(state_size, goal_size, num_agents, xy_scale, z_scale).to(device)
        self.value_net = ManagerValueNetwork(state_size).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def policy(self, x):
        return self.policy_net(x)
    
    def value(self, x):
        return self.value_net(x)

class WorkerAgent:
    def __init__(self, state_size, action_size, lr=5e-4):
        self.policy_net = WorkerNetwork(state_size, action_size).to(device)
        self.value_net = WorkerValueNetwork(state_size).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def policy(self, x):
        return self.policy_net(x)
    
    def value(self, x):
        return self.value_net(x)

# =============================================================================
# Helper Functions for Observation & Reward Processing
# =============================================================================

def extract_position(obs):
    """Extract the agent’s own position (first 3 numbers) from an observation."""
    return np.array(obs[:3])

def build_manager_global_state(obs_dict):
    """
    Build a global state by concatenating each agent's position (3 numbers per agent)
    followed by the target’s position (3 numbers).
    """
    agent_keys = sorted(obs_dict.keys())
    target = np.array(obs_dict[agent_keys[0]][3:6])
    features = []
    for agent in agent_keys:
        features.extend(extract_position(obs_dict[agent]))
    features.extend(target)
    return np.array(features)

def build_worker_state(obs_dict, agent_id):
    """Build a worker’s state (its own position)."""
    return extract_position(obs_dict[agent_id])

def get_global_worker_state(worker_agent):
    return {
        "policy_net": copy.deepcopy(worker_agent.policy_net.state_dict()),
        "value_net": copy.deepcopy(worker_agent.value_net.state_dict())
    }

def discount_rewards(rewards, gamma):
    """Compute the discounted return for a list of rewards."""
    discounted = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted.insert(0, R)
    return discounted

def curriculum_weight(episode, curriculum_threshold=1000):
    """Linearly increase from 0 to 1 over curriculum_threshold episodes."""
    return min(1.0, episode / curriculum_threshold)

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# =============================================================================
# Environment Randomization Helpers
# =============================================================================

def random_target_speed(min_speed=0.1, max_speed=0.5):
    return np.random.uniform(min_speed, max_speed)

def random_env_size(min_size, max_size):
    return np.random.randint(min_size, max_size + 1)

def random_init_flying_pos_with_min_separation(num_drones, xy_range, z_range, min_sep):
    positions = []
    attempts = 0
    max_attempts = 1000
    while len(positions) < num_drones:
        candidate = np.array([
            np.random.uniform(xy_range[0], xy_range[1]),
            np.random.uniform(xy_range[0], xy_range[1]),
            np.random.uniform(z_range[0], z_range[1])
        ])
        if all(np.linalg.norm(candidate[:2] - pos[:2]) >= min_sep for pos in positions):
            positions.append(candidate)
        attempts += 1
        if attempts > max_attempts:
            raise ValueError("Unable to sample positions with specified minimum separation.")
    return np.array(positions)

def random_init_target_location_with_min_separation(xy_range, z_range, drone_positions, min_sep):
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
            raise ValueError("Unable to sample target location with specified minimum separation.")

def randomize_catch_parameters(num_drones=2,
                               env_size_range=(2, 5),
                               z_range_flying=(1, 2),
                               z_range_target=(0.2, 3),
                               min_sep=2):
    env_size = random_env_size(env_size_range[0], env_size_range[1])
    xy_range = (-env_size, env_size)
    init_flying_pos = random_init_flying_pos_with_min_separation(num_drones, xy_range, z_range_flying, min_sep)
    init_target_location = random_init_target_location_with_min_separation(xy_range, z_range_target, init_flying_pos, min_sep)
    return {
        "init_flying_pos": init_flying_pos,
        "init_target_location": init_target_location,
        "env_size": env_size
    }

# =============================================================================
# Generalized Advance Estimation Helper Function
# =============================================================================

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: list of rewards at each timestep
    values: list of value estimates at each timestep + 1 final value (for bootstrap)
    gamma: discount factor
    lam: GAE parameter
    
    returns: list of advantage estimates, same length as rewards
    """
    advantages = []
    gae = 0
    # We assume values has one extra element: values[t+1] is the "bootstrap" for the last step
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

# =============================================================================
# Vectorized Consistency Loss Helper Function
# =============================================================================

def compute_vectorized_consistency_loss(manager_goals_list, obs_history, agent_keys):
    """
    Compute the consistency loss in a vectorized manner.
    
    Args:
        manager_goals_list (list of torch.Tensor): Each tensor is of shape (num_agents, goal_size)
            and corresponds to the manager’s predicted goals at one timestep.
        obs_history (list of dict): Each element is a dictionary of observations for one timestep.
            Each observation is assumed to be a numpy array where the first 3 elements are the
            agent’s position and elements 3:6 represent the target’s position.
        agent_keys (list of str): List of keys corresponding to each agent.
    
    Returns:
        torch.Tensor: The scalar consistency loss.
    """
    # Stack predicted manager goals: shape (T, num_agents, goal_size)
    predicted_goals_tensor = torch.stack(manager_goals_list, dim=0)
    
    # For each timestep, extract the true target from the first agent's observation.
    # Each true target is a 3-element vector.
    true_targets = torch.stack([
        torch.FloatTensor(obs[agent_keys[0]][3:6])
        for obs in obs_history
    ], dim=0)  # shape: (T, 3)
    
    # Expand true targets to match the shape (T, num_agents, 3)
    true_targets_expanded = true_targets.unsqueeze(1).expand(-1, len(agent_keys), -1)
    
    # For each timestep and each agent, extract the worker's position (first 3 elements).
    worker_positions = torch.stack([
        torch.stack([torch.FloatTensor(obs[agent][:3]) for agent in agent_keys], dim=0)
        for obs in obs_history
    ], dim=0)  # shape: (T, num_agents, 3)
    
    # The desired goal is the vector difference between the true target and the worker position.
    desired_goals = true_targets_expanded - worker_positions  # shape: (T, num_agents, 3)
    
    # Compute the mean squared error between predicted goals and desired goals.
    consistency_loss = F.mse_loss(predicted_goals_tensor, desired_goals, reduction='mean')
    
    return consistency_loss

# =============================================================================
# Scheduled FedProx Coefficient Helper Function
# =============================================================================

def scheduled_fedprox_mu(episode, schedule_threshold, mu_max=0.1):
    """
    Linearly schedule the FedProx coefficient from 0 to mu_max over schedule_threshold episodes.
    
    Args:
        episode: Current global episode number.
        schedule_threshold: Number of episodes over which to reach mu_max.
        mu_max: The final (maximum) FedProx coefficient.
        
    Returns:
        A scheduled FedProx coefficient.
    """
    
    return min(mu_max, (episode / schedule_threshold) * mu_max)

# =============================================================================
# FedProx Helper Function
# =============================================================================

def compute_fedprox_term(worker_agent, global_state, mu):
    """
    Compute the FedProx proximal term for a worker agent.
    
    Args:
        worker_agent: The WorkerAgent instance (shared by agents in a client).
        global_state: The global snapshot (a dict with keys 'policy_net' and 'value_net').
        mu: The proximal term coefficient.
        
    Returns:
        A scalar proximal loss term.
    """
    prox_term = 0.0
    # For the policy network.
    for name, param in worker_agent.policy_net.named_parameters():
        global_param = global_state["policy_net"][name]
        prox_term += (mu / 2.0) * torch.norm(param - global_param) ** 2
    # For the value network.
    for name, param in worker_agent.value_net.named_parameters():
        global_param = global_state["value_net"][name]
        prox_term += (mu / 2.0) * torch.norm(param - global_param) ** 2
    return prox_term

# =============================================================================
# Feudal Episode Training Function
# =============================================================================

def train_feudal_episode(manager, workers, env, episode,
                         curriculum_threshold=1000, gamma=0.995, entropy_coef=0.01,
                         consistency_lambda=0.01, intrinsic_coef=0.1,
                         global_worker_state=None, mu=1e-4, schedule_threshold=7500):
    """
    Run one feudal RL episode for training.
      - The manager computes a goal for each worker based on a global state.
      - Each worker blends the manager’s goal with the true target (using a curriculum weight)
        and samples an action.
      - Extrinsic and intrinsic rewards are computed.
      - Actor-critic losses are computed and both manager and worker networks are updated.
      - If a global_worker_state is provided, a FedProx term is added to the worker loss.
    """
    # Logs for manager and workers.
    manager_log_probs_list = []
    manager_rewards = []
    manager_entropies_list = []
    manager_values = []
    
    agent_keys = sorted(workers.keys())
    worker_log_probs = {agent: [] for agent in agent_keys}
    worker_rewards = {agent: [] for agent in agent_keys}
    worker_entropies = {agent: [] for agent in agent_keys}
    worker_values = {agent: [] for agent in agent_keys}
    
    # For consistency loss.
    manager_goals_list = []
    obs_history = []
    
    obs_dict, _ = env.reset()
    done = False
    total_episode_reward = 0

    while not done:
        obs_history.append(obs_dict)
        # Manager step.
        global_state = build_manager_global_state(obs_dict)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        m_mean, m_std = manager.policy(global_state_tensor)
        m_value = manager.value(global_state_tensor)
        manager_values.append(m_value)
        m_dist = torch.distributions.Normal(m_mean, m_std)
        m_sample = m_dist.sample().squeeze(0)  # shape: (num_workers, goal_size)
        m_log_prob = m_dist.log_prob(m_sample.unsqueeze(0)).sum(dim=2).squeeze(0)
        m_entropy = m_dist.entropy().sum(dim=2).squeeze(0)
        manager_log_probs_list.append(m_log_prob)
        manager_entropies_list.append(m_entropy)
        manager_goals_list.append(m_sample)
        
        # Worker step.
        worker_goals = {}
        actions = {}
        for i, agent in enumerate(agent_keys):
            local_state = build_worker_state(obs_dict, agent)
            cw = curriculum_weight(episode, curriculum_threshold)
            true_target = np.array(obs_dict[agent][3:6])
            manager_goal = m_sample[i].detach().cpu().numpy()
            worker_goal = (1 - cw) * true_target + cw * manager_goal
            worker_goals[agent] = worker_goal
            
            worker_input = np.concatenate([local_state, worker_goal])
            worker_input_tensor = torch.FloatTensor(worker_input).unsqueeze(0).to(device)
            w_value = workers[agent].value(worker_input_tensor)
            worker_values[agent].append(w_value)
            w_mean, w_std = workers[agent].policy(worker_input_tensor)
            w_dist = torch.distributions.Normal(w_mean, w_std)
            w_sample = w_dist.sample()
            action = w_sample.squeeze(0).detach().cpu().numpy()
            log_prob = w_dist.log_prob(w_sample).sum(dim=1)
            entropy = w_dist.entropy().sum(dim=1)
            worker_log_probs[agent].append(log_prob)
            worker_entropies[agent].append(entropy)
            actions[agent] = action
        
        # Step environment.
        next_obs_dict, reward_dict, terminated, truncated, _ = env.step(actions)
        done = all(terminated[agent] or truncated[agent] for agent in agent_keys)
        
        # Compute intrinsic rewards.
        global_step_reward = 0
        for agent in agent_keys:
            worker_pos_next = extract_position(next_obs_dict[agent])
            intrinsic_reward = -intrinsic_coef * np.linalg.norm(worker_pos_next - worker_goals[agent])
            intrinsic_reward = reward_dict[agent] + intrinsic_reward
            worker_rewards[agent].append(intrinsic_reward)
            global_step_reward += intrinsic_reward
        
        manager_rewards.append(global_step_reward)
        total_episode_reward += global_step_reward
        obs_dict = next_obs_dict

    # Check if target was captured.
    first_key = list(obs_dict.keys())[0]
    target_pos = obs_dict[first_key][3:6]
    caught_train = any(
        np.linalg.norm(extract_position(obs_dict[agent]) - target_pos) < CLOSENESS_THRESHOLD
        for agent in obs_dict
    )

    # Manager loss (actor-critic) using GAE.
    # Append a bootstrap value (0) for the final state.
    manager_values.append(torch.zeros(1, 1, device=device))
    manager_values_tensor = torch.cat(manager_values, dim=0).squeeze(1)
    # Compute GAE returns; note we pass the value estimates as a list of numbers.
    m_returns = compute_gae(manager_rewards, [v.item() for v in manager_values_tensor], gamma, lam=0.95)
    m_returns = torch.FloatTensor(m_returns)
    # Compute advantages using the first T values of manager_values_tensor.
    m_advantages = m_returns - manager_values_tensor[:-1].detach()

    
    manager_actor_loss = 0
    manager_critic_loss = 0
    for t in range(len(m_returns)):
        step_log_prob = manager_log_probs_list[t].sum()
        step_entropy = manager_entropies_list[t].sum()
        manager_actor_loss += -step_log_prob * m_advantages[t] - entropy_coef * step_entropy
        manager_critic_loss += F.mse_loss(manager_values_tensor[t].unsqueeze(0), m_returns[t].unsqueeze(0))
    value_coef = 0.5
    manager_loss = manager_actor_loss + value_coef * manager_critic_loss

    # Worker loss.
    worker_loss_total = 0
    worker_actor_loss_total = 0
    worker_critic_loss_total = 0
    
    for agent in agent_keys:
        # Append a bootstrap value (0) for the final worker value.
        worker_values[agent].append(torch.zeros(1, 1))
        w_values_tensor = torch.cat(worker_values[agent], dim=0).squeeze(1)
        # Compute GAE returns; pass the value estimates as a list of numbers.
        r = compute_gae(worker_rewards[agent], [v.item() for v in w_values_tensor], gamma, lam=0.95)
        r = torch.FloatTensor(r)
        # Compute advantages using the first T values.
        advantages = r - w_values_tensor[:-1].detach()
        agent_actor_loss = 0
        agent_critic_loss = 0
        for t in range(len(r)):
            agent_actor_loss += -worker_log_probs[agent][t] * advantages[t] - entropy_coef * worker_entropies[agent][t]
            agent_critic_loss += F.mse_loss(w_values_tensor[t].unsqueeze(0), r[t].unsqueeze(0))
        worker_actor_loss_total += agent_actor_loss
        worker_critic_loss_total += agent_critic_loss
        
    worker_loss_total = worker_actor_loss_total + value_coef * worker_critic_loss_total
    
    
    # If a global worker state is provided, add the FedProx proximal term.
    if global_worker_state is not None:
        # Compute a scheduled FedProx coefficient based on the current episode.
        scheduled_mu = scheduled_fedprox_mu(episode, schedule_threshold=schedule_threshold, mu_max=mu)
        prox_loss = compute_fedprox_term(workers[agent_keys[0]], global_worker_state, scheduled_mu)
        worker_loss_total = worker_loss_total + prox_loss

    # Consistency loss (optional) - vectorized implementation.
    consistency_loss = 0
    if episode >= curriculum_threshold:
        consistency_loss = compute_vectorized_consistency_loss(manager_goals_list, obs_history, agent_keys)



    global_loss = manager_loss + worker_loss_total + consistency_lambda * consistency_loss
    
    # Gradient clipping max norm value
    max_grad_norm = 5.0  

    # Update worker networks.
    for agent in agent_keys:
        workers[agent].policy_optimizer.zero_grad()
        workers[agent].value_optimizer.zero_grad()
    worker_loss_total.backward(retain_graph=True)
    
    # Apply gradient clipping to each worker network
    for agent in agent_keys:
        torch.nn.utils.clip_grad_norm_(workers[agent].policy_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(workers[agent].value_net.parameters(), max_grad_norm) 
    
    for agent in agent_keys:
        workers[agent].policy_optimizer.step()
        workers[agent].value_optimizer.step()
    
    # Update manager network.
    manager.policy_optimizer.zero_grad()
    manager.value_optimizer.zero_grad()
    manager_loss.backward()
    
    # Clip gradients for the manager networks
    torch.nn.utils.clip_grad_norm_(manager.policy_net.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(manager.value_net.parameters(), max_grad_norm)
    
    manager.policy_optimizer.step()
    manager.value_optimizer.step()
    
    return total_episode_reward, manager_loss.item(), worker_loss_total.item(), global_loss.item(), caught_train

# =============================================================================
# Federated Averaging for Worker Networks
# =============================================================================

def federated_average_workers(worker_agents_list):
    """
    Given a list of WorkerAgent objects (one per client), average the parameters
    of both the policy and value networks.
    """
    num_clients = len(worker_agents_list)
    # Average policy_net parameters.
    global_policy_state = {}
    for key in worker_agents_list[0].policy_net.state_dict().keys():
        global_policy_state[key] = sum(agent.policy_net.state_dict()[key] for agent in worker_agents_list) / num_clients
    # Average value_net parameters.
    global_value_state = {}
    for key in worker_agents_list[0].value_net.state_dict().keys():
        global_value_state[key] = sum(agent.value_net.state_dict()[key] for agent in worker_agents_list) / num_clients
    # Update each worker agent.
    for agent in worker_agents_list:
        agent.policy_net.load_state_dict(global_policy_state)
        agent.value_net.load_state_dict(global_value_state)

# =============================================================================
# Federated FeudRL Training Loop
# =============================================================================

def federated_feudal_training(num_clients=4, comm_rounds=100, episodes_per_round=10,
                              curriculum_threshold=100, gamma=0.995, entropy_coef=0.01,
                              lr=5e-4, schedule_threshold=750):
    """
    Run federated feudal RL training.
      - A single global manager is created.
      - Each client has its own local environment and a shared WorkerAgent (used for both agents).
      - For each communication round, every client runs several local episodes.
      - After each round, the local worker networks are averaged (federated averaging).
    """
    agents_per_client = 2  # 2 agents per client (using a shared worker network)
    client_worker_agents = []  # One WorkerAgent per client
    client_envs = []
    
    # Create a global manager.
    # The manager state: (3 numbers per worker + 3 for target) 
    manager_state_size = 3 * agents_per_client + 3  
    goal_size = 3  # Predicted goal (target position)
    worker_local_state_size = 3  # Each worker observes its own position.
    worker_state_size = worker_local_state_size + goal_size  # Worker input dimension = 6.
    action_size = 3
    
    # For initializing parameters.
    params = randomize_catch_parameters(num_drones=agents_per_client)
    global_manager = ManagerAgent(manager_state_size, goal_size, num_agents=agents_per_client,
                                  xy_scale=params["env_size"], z_scale=3, lr=lr)
    
    # Create LR schedulers for the manager's optimizers.
    manager_policy_scheduler = torch.optim.lr_scheduler.StepLR(
        global_manager.policy_optimizer, step_size=100, gamma=0.9)
    manager_value_scheduler = torch.optim.lr_scheduler.StepLR(
        global_manager.value_optimizer, step_size=100, gamma=0.9)    
    
    # Create clients.
    for _ in range(num_clients):
        params = randomize_catch_parameters(num_drones=agents_per_client)
        env = Catch(
            drone_ids=np.arange(agents_per_client),
            render_mode=None,
            init_flying_pos=params["init_flying_pos"],
            init_target_location=params["init_target_location"],
            size=params["env_size"],
            target_speed=random_target_speed()
        )
        client_envs.append(env)
        # Create a single WorkerAgent per client (shared by both agents).
        worker_agent = WorkerAgent(worker_state_size, action_size, lr=lr)
        # Build a dictionary mapping each agent id to the same worker_agent.
        # Attach LR schedulers to the worker agent.
        worker_agent.policy_scheduler = torch.optim.lr_scheduler.StepLR(
            worker_agent.policy_optimizer, step_size=100, gamma=0.9)
        worker_agent.value_scheduler = torch.optim.lr_scheduler.StepLR(
            worker_agent.value_optimizer, step_size=100, gamma=0.9)
        worker_dict = {f'agent_{i}': worker_agent for i in range(agents_per_client)}
        client_worker_agents.append(worker_dict)
    
    # For logging.
    round_rewards = []
    start_time = time.time()
    
    # Main federated training loop.
    for comm_round in range(comm_rounds):
        round_rewards_clients = []
        # For each client, run episodes_per_round local episodes.
        for client_idx in range(num_clients):
            env = client_envs[client_idx]
            workers = client_worker_agents[client_idx]
            client_rewards = []
            global_worker_state = get_global_worker_state(list(workers.values())[0])
            # Run local episodes.
            for ep in range(episodes_per_round):
                # Use a unique global episode counter if desired; here we use ep.
                ep_reward, m_loss, w_loss, g_loss, caught = train_feudal_episode(
                    global_manager, 
                    workers, 
                    env, 
                    episode=comm_round * episodes_per_round + ep,
                    curriculum_threshold=curriculum_threshold, 
                    gamma=gamma, 
                    entropy_coef=entropy_coef,
                    global_worker_state=global_worker_state,
                    schedule_threshold=schedule_threshold
                )
                client_rewards.append(ep_reward)
            round_rewards_clients.append(np.mean(client_rewards))
        avg_round_reward = np.mean(round_rewards_clients)
        round_rewards.append(avg_round_reward)
        elapsed = (time.time() - start_time) / 60
        print(f"\rRound {comm_round+1}/{comm_rounds} - Avg Reward: {avg_round_reward:.2f} - Curriculum weight {curriculum_weight((comm_round+1)*episodes_per_round, curriculum_threshold):.2f} | Time: {elapsed:.2f} min", end="", flush=True)
        
        # Federated averaging: average the worker networks across clients.
        # First, extract the WorkerAgent from each client (they share one per client).
        worker_agents_list = [list(client_worker_agents[i].values())[0] for i in range(num_clients)]
        federated_average_workers(worker_agents_list)
        
        # Step the learning rate schedulers.
        # For each worker agent.
        for worker_dict in client_worker_agents:
            worker = list(worker_dict.values())[0]
            worker.policy_scheduler.step()
            worker.value_scheduler.step()
        # For the manager.
        manager_policy_scheduler.step()
        manager_value_scheduler.step()
        
        # Optionally, print the current learning rate:
        current_lr = global_manager.policy_optimizer.param_groups[0]['lr']
        print(f" | Current LR: {current_lr:.6f}", end="")        
        
        
    # Close all client environments.
    for env in client_envs:
        env.close()
    
    # Plot training progress.
    plt.figure(figsize=(8, 5))
    plt.plot(round_rewards)
    plt.xlabel('Communication Round')
    plt.ylabel('Average Episode Reward')
    plt.title('Federated FeudRL Training Progress')
    plt.grid(True)
    plt.show()
    
    return round_rewards, global_manager, client_worker_agents

# =============================================================================
# (Optional) Testing the Trained Federated FeudRL Model
# =============================================================================

def execute_feudal_episode(manager, workers, env):
    """
    Execute one feudal episode in render mode using the provided manager and workers.
    """
    obs_dict, _ = env.reset()
    agent_keys = sorted(workers.keys())
    done = False

    while not done:
        env.render()
        pygame.event.pump()
        pygame.time.wait(100)
        
        # Manager step.
        global_state = build_manager_global_state(obs_dict)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        m_mean, m_std = manager.policy(global_state_tensor)
        m_dist = torch.distributions.Normal(m_mean, m_std)
        m_sample = m_dist.sample().squeeze(0)
        print("Manager goals:", m_sample.cpu().numpy())
        
        actions = {}
        for i, agent in enumerate(agent_keys):
            local_state = build_worker_state(obs_dict, agent)
            manager_goal = m_sample[i].detach().cpu().numpy()
            worker_input = np.concatenate([local_state, manager_goal])
            worker_input_tensor = torch.FloatTensor(worker_input).unsqueeze(0).to(device)
            w_mean, w_std = workers[agent].policy(worker_input_tensor)
            w_dist = torch.distributions.Normal(w_mean, w_std)
            w_sample = w_dist.sample()
            action = w_sample.squeeze(0).detach().cpu().numpy()
            actions[agent] = action

        next_obs_dict, reward_dict, terminated, truncated, _ = env.step(actions)
        done = all(terminated[agent] or truncated[agent] for agent in agent_keys)
        obs_dict = next_obs_dict

    # Check for capture.
    first_key = list(obs_dict.keys())[0]
    target_pos = obs_dict[first_key][3:6]
    caught = any(
        np.linalg.norm(extract_position(obs_dict[agent]) - target_pos) < CLOSENESS_THRESHOLD
        for agent in obs_dict
    )
    env.close()
    return caught

def test_feudrl_model(manager, client_worker_agents, test_rounds=5):
    """
    Test the trained federated feudal RL model using one of the client environments.
    """
    # Use the worker network from the first client.
    workers = client_worker_agents[0]
    success_count = 0
    for _ in range(test_rounds):
        params = randomize_catch_parameters(num_drones=len(workers))
        test_env = Catch(
            drone_ids=np.arange(len(workers)),
            render_mode="human",
            init_flying_pos=params["init_flying_pos"],
            init_target_location=params["init_target_location"],
            size=params["env_size"],
            target_speed=random_target_speed()
        )
        caught = execute_feudal_episode(manager, workers, test_env)
        success_count += int(caught)
    success_rate = success_count / test_rounds * 100
    print(f"\nTest Success Rate: {success_rate:.2f}% over {test_rounds} episodes.")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    lr = 5e-4 # 5e-4 1e-5
    entropy_coef = 0.01
    gamma = 0.95
    comm_rounds = 1000
    episodes_per_round = 10
    total_eps = comm_rounds*episodes_per_round
    curriculum_rate = 0.5
    fedprox_rate = 0.75
    
    rewards, global_manager, client_worker_agents = federated_feudal_training(
        num_clients=4,
        comm_rounds=comm_rounds,
        episodes_per_round=episodes_per_round,
        curriculum_threshold=curriculum_rate*total_eps,
        gamma=gamma,
        entropy_coef=entropy_coef,
        lr=lr,
        schedule_threshold=total_eps*fedprox_rate
        
    )
    
    # Optionally, test the trained model.
    test_feudrl_model(global_manager, client_worker_agents, test_rounds=10)


