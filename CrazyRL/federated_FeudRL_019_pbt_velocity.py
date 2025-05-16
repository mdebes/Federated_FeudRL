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
import optuna
import os
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from codecarbon import EmissionsTracker

# Import the Catch environment and closeness constant.
from crazy_rl.multi_agent.numpy.catch.catch import Catch

# Set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
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
        # Modified code: allows z_mean to be in [-z_scale, z_scale] from 0.2 to 3 <- removed the 0.2
        z_mean = (torch.tanh(raw_mean[:, :, 2]) + 1) * self.z_scale
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
    def __init__(self, state_size, goal_size, num_agents, xy_scale, z_scale, lr=5e-4, hidden_size=128):
        self.policy_net = ManagerNetwork(state_size, goal_size, num_agents, xy_scale, z_scale, hidden_size).to(device)
        self.value_net = ManagerValueNetwork(state_size, hidden_size).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def policy(self, x):
        return self.policy_net(x)
    
    def value(self, x):
        return self.value_net(x)

class WorkerAgent:
    def __init__(self, state_size, action_size, lr=5e-4, hidden_size=128):
        self.policy_net = WorkerNetwork(state_size, action_size, hidden_size).to(device)
        self.value_net = WorkerValueNetwork(state_size, hidden_size).to(device)
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

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# =============================================================================
# Environment Randomization Helpers
# =============================================================================

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
                               env_size=5,
                               z_range_flying=(1, 2),
                               z_range_target=(0.2, 2.8),
                               min_sep=2):
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

def train_feudal_episode(manager, 
                         workers,
                         env,
                         episode,
                         gamma=0.995,
                         entropy_coef=0.01,
                         global_worker_state=None,
                         mu=1e-4,
                         lam=0.95,
                         manager_interval=3):
    """
    Run one feudal RL episode for training.
      - The manager computes a goal for each worker based on a global state.
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
       
    obs_dict, _ = env.reset()
    done = False
    last_target_pos = False
    total_episode_reward = 0
    steps = 0
    last_info = {}
    
    # For the manager’s temporally extended goal.
    accumulated_manager_reward = 0.0  # Accumulates reward between manager updates

    while not done:
        steps += 1
        
        # On manager update steps, compute a new goal.
        if (steps - 1) % manager_interval == 0:
            # If not the first manager update, store the accumulated reward
            if steps > 1:
                manager_rewards.append(accumulated_manager_reward)
                accumulated_manager_reward = 0.0
            # Manager step: compute new goal from the current global state.
            pos_state = build_manager_global_state(obs_dict) # Current position to calculate velocity
            # Extract & compute target velocity
            current_target_pos = pos_state[-3:]  # Calculate velocity
            if last_target_pos is None:
                tgt_vel = np.zeros(3, dtype=current_target_pos.dtype)
            else:
                tgt_vel = current_target_pos - last_target_pos
            last_target_pos = current_target_pos.copy()            
                       
            # Concatenate positions + target velocity
            global_state = np.concatenate([pos_state, tgt_vel])  # length = 3*N + 3 + 3

            global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
            m_mean, m_std = manager.policy(global_state_tensor)
            m_value = manager.value(global_state_tensor)
            manager_values.append(m_value)
            # Sample a goal from the manager’s distribution.
            m_dist = torch.distributions.Normal(m_mean, m_std)
            m_sample = m_dist.sample().squeeze(0)  # shape: (num_workers, goal_size)
            # Store manager log probability, entropy, and value.
            m_log_prob = m_dist.log_prob(m_sample.unsqueeze(0)).sum(dim=2).squeeze(0)
            m_entropy = m_dist.entropy().sum(dim=2).squeeze(0)
            manager_log_probs_list.append(m_log_prob)
            manager_entropies_list.append(m_entropy)
        
        # Worker step.
        actions = {}
        for i, agent in enumerate(agent_keys):
            local_state = build_worker_state(obs_dict, agent)
            worker_goal = m_sample[i].detach().cpu().numpy()
            
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
        next_obs_dict, reward_dict, terminated, truncated, info = env.step(actions)
        last_info = info  # keep the info from the last step
        done = all(terminated[agent] or truncated[agent] for agent in agent_keys)
        
        # For workers: record rewards as before.
        global_step_reward = 0
        for agent in agent_keys:
            worker_rewards[agent].append(reward_dict[agent])
            global_step_reward += reward_dict[agent]
        total_episode_reward += global_step_reward
        
        # Also accumulate this reward for the manager update.
        accumulated_manager_reward += global_step_reward
        
        obs_dict = next_obs_dict        
        
    # At episode end, append any remaining accumulated manager reward.
    manager_rewards.append(accumulated_manager_reward)
    
    # Check if target was captured.
    caught_train = any(last_info.get("caught", {}).values())

    # ---- Manager Loss (Actor-Critic with GAE) ----
    # Append a bootstrap value (0) for the final state.
    manager_values.append(torch.zeros(1, 1, device=device))
    manager_values_tensor = torch.cat(manager_values, dim=0).squeeze(1)
    # Compute GAE returns; note we pass the value estimates as a list of numbers.
    m_returns = compute_gae(manager_rewards, [v.item() for v in manager_values_tensor], gamma, lam)
    m_returns = torch.FloatTensor(m_returns).to(device)
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

    # ---- Worker Loss (Actor-Critic) ----
    worker_loss_total = 0
    worker_actor_loss_total = 0
    worker_critic_loss_total = 0
    
    for agent in agent_keys:
        # Append a bootstrap value (0) for the final worker value.
        worker_values[agent].append(torch.zeros(1, 1, device=device))
        w_values_tensor = torch.cat(worker_values[agent], dim=0).squeeze(1)
        # Compute GAE returns; pass the value estimates as a list of numbers.
        r = compute_gae(worker_rewards[agent], [v.item() for v in w_values_tensor], gamma, lam)
        r = torch.FloatTensor(r).to(device)
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
        # Compute FedProx coefficient
        prox_loss = compute_fedprox_term(workers[agent_keys[0]], global_worker_state, mu)
        worker_loss_total = worker_loss_total + prox_loss

    global_loss = manager_loss + worker_loss_total
    
    # ---- Backpropagation ----
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
    
    # Return metrics.
    return total_episode_reward, manager_loss.item(), worker_loss_total.item(), global_loss.item(), caught_train, steps

# =============================================================================
# Federated FeudRL Training Loop
# =============================================================================

def federated_feudal_training(num_clients=4,
                              m_hidden_size=128,
                              w_hidden_size=128,
                              comm_rounds=100, 
                              episodes_per_round=10,
                              gamma=0.995,
                              entropy_coef=0.01,
                              manager_lr=5e-4,
                              worker_lr=1e-4,
                              mu=0.1,
                              lam=0.1):
    """
    Run federated feudal RL training.
      - A single global manager is created.
      - Each client has its own local environment and a shared WorkerAgent (used for both agents).
      - For each communication round, every client runs several local episodes.
      - After each round, the local worker networks are averaged (federated averaging).
    """
    agents_per_client = 2  # 2 agents per client (using a shared worker network)
    client_worker_agents = []  # One WorkerAgent per client
    
    # Create a global manager.
    # The manager state: (3 numbers per worker + 3 for target + 3 for target velocity) ) 
    manager_state_size = 3 * agents_per_client + 3  + 3 
    goal_size = 3  # Predicted goal (target position)
    worker_local_state_size = 3  # Each worker observes its own position.
    worker_state_size = worker_local_state_size + goal_size  # Worker input dimension = 6.
    action_size = 3
    
    # For initializing parameters.
    params = randomize_catch_parameters(num_drones=agents_per_client)
    global_manager = ManagerAgent(manager_state_size, goal_size, num_agents=agents_per_client,
                                  xy_scale=params["env_size"], z_scale=1.5, lr=manager_lr, hidden_size=m_hidden_size)
    
    # Create LR schedulers for the manager's optimizers.
    manager_policy_scheduler = torch.optim.lr_scheduler.StepLR(
        global_manager.policy_optimizer, step_size=100, gamma=gamma)
    manager_value_scheduler = torch.optim.lr_scheduler.StepLR(
        global_manager.value_optimizer, step_size=100, gamma=gamma)   
    
   
    # Create empty client_envs and worker agents (but defer env-instantiation until each episode)
    for _ in range(num_clients):
        # we'll build each episode's env on the fly, so just store the worker dict
        worker_agent = WorkerAgent(worker_state_size, action_size, lr=worker_lr, hidden_size=w_hidden_size)
        worker_agent.policy_scheduler = torch.optim.lr_scheduler.StepLR(
            worker_agent.policy_optimizer, step_size=100, gamma=gamma)
        worker_agent.value_scheduler = torch.optim.lr_scheduler.StepLR(
            worker_agent.value_optimizer, step_size=100, gamma=gamma)
        client_worker_agents.append({f'agent_{i}': worker_agent for i in range(agents_per_client)})

    
    # Early stopping parameters.
    stop_threshold = 95.0  # e.g., 95% capture rate.
    patience = 5           # number of consecutive rounds required.
    high_capture_counter = 0

    # For logging.
    round_rewards = []
    capture_rates = []
    round_steps = []
    start_time = time.time()


    
    # Main federated training loop.
    for comm_round in range(comm_rounds):
        round_rewards_clients = []
        client_capture_rates = []  # List to hold capture rate for each client
        round_steps_clients = [] 
        
        # For each client, run episodes_per_round local episodes.
        for client_idx in range(num_clients):
            workers = client_worker_agents[client_idx]
            client_rewards = []
            capture_count = 0  # Count how many episodes the target is captured
            global_worker_state = get_global_worker_state(list(workers.values())[0])
            
            # Run local episodes (with fresh env each time)
            for ep in range(episodes_per_round):
                # --- begin per-episode randomization ---
                params = randomize_catch_parameters(num_drones=agents_per_client)
                env = Catch(
                    drone_ids=np.arange(agents_per_client),
                    render_mode=None,
                    init_flying_pos=params["init_flying_pos"],
                    init_target_location=params["init_target_location"],
                    size=params["env_size"],
                    target_speed=0.5
                )
                # --- end per-episode randomization ---
                try:
                    ep_reward, m_loss, w_loss, g_loss, caught, steps = train_feudal_episode(
                        global_manager,
                        workers,
                        env,
                        episode=comm_round * episodes_per_round + ep,
                        gamma=gamma,
                        entropy_coef=entropy_coef,
                        global_worker_state=global_worker_state,
                        mu=mu,
                        lam=lam,
                        manager_interval=3
                    )
                # Close env for the episode so a new one can be initialized
                finally:
                    env.close()
                client_rewards.append(ep_reward)
                # If the target was caught in this episode, count it.
                if caught:
                    capture_count += 1
        
            
            # Compute this client's capture rate.
            client_capture_rate = (capture_count / episodes_per_round) * 100.0
            client_capture_rates.append(client_capture_rate)
            round_rewards_clients.append(np.mean(client_rewards))  
            round_steps_clients.append(steps)
            
        # Compute average metrics over clients.
        avg_round_reward = np.mean(round_rewards_clients)
        avg_round_capture_rate = np.mean(client_capture_rates)
        avg_round_steps = np.mean(round_steps_clients) 
        
        round_rewards.append(avg_round_reward)
        capture_rates.append(avg_round_capture_rate) 
        round_steps.append(avg_round_steps)
        
        elapsed = (time.time() - start_time) / 60
        print(f"\rRound {comm_round+1}/{comm_rounds} - Avg Reward: {avg_round_reward:.2f} - Capture Rate: {avg_round_capture_rate:.2f} | Time: {elapsed:.2f} min", end="", flush=True)

        # Early stopping check.
        if avg_round_capture_rate >= stop_threshold:
            high_capture_counter += 1
            if high_capture_counter >= patience:
                print(f"\nEarly stopping triggered after {high_capture_counter} consecutive rounds with capture rate >= {stop_threshold}% at round {comm_round+1}/{comm_rounds}")
                break
        else:
            high_capture_counter = 0        

        # === Best-in-Class update: broadcast the single best client’s weights ===
        # pick the client with highest average reward this round
        best_idx    = int(np.argmax(round_rewards_clients))
        best_worker = list(client_worker_agents[best_idx].values())[0]

        # grab its policy & value state dicts
        best_policy_state = best_worker.policy_net.state_dict()
        best_value_state  = best_worker.value_net.state_dict()

        # overwrite every client’s worker nets with the elite’s weights
        for client_dict in client_worker_agents:
            w = list(client_dict.values())[0]
            w.policy_net.load_state_dict(best_policy_state)
            w.value_net .load_state_dict(best_value_state)
        
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
        current_manager_lr = global_manager.policy_optimizer.param_groups[0]['lr']
        any_worker = list(client_worker_agents[0].values())[0]
        current_worker_lr = any_worker.policy_optimizer.param_groups[0]['lr']
        print(f" | Manager LR: {current_manager_lr:.6f} | Worker LR: {current_worker_lr:.6f} ", end="")  
        
    # Save results

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    np.savez(
        f"training_metrics__pbt_{timestamp}.npz",
        round_rewards=round_rewards,
        capture_rates=capture_rates
    )    
        
        
    # Plotting the training progress
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.plot(round_rewards, color=color, label='Avg Reward')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Average Episode Reward', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # create a second y-axis sharing the same x-axis
    color = 'tab:brown'
    ax2.plot(capture_rates, color=color, label='Capture Rate')
    ax2.set_ylabel('Capture Rate (%)', color=color)
    ax2.set_ylim(0, 100)  # ensure the right axis is from 0 to 100
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Federated FeudRL PBT A2C Training Progress')
    fig.tight_layout()  # adjust layout to prevent overlapping labels
    plt.grid(True)
    plt.show()
    
    # Graphic for average steps per communication round.
    fig_steps, ax_steps = plt.subplots(figsize=(8, 5))
    ax_steps.plot(round_steps, color='tab:green', label='Avg Steps per Episode')
    ax_steps.set_xlabel('Communication Round')
    ax_steps.set_ylabel('Average Steps per Episode')
    ax_steps.set_title('Average Steps per Communication Round')
    ax_steps.grid(True)
    plt.legend()
    plt.show()
    
    return round_rewards, global_manager, client_worker_agents, capture_rates

# =============================================================================
# Testing the Trained Federated FeudRL Model
# =============================================================================

def visualize_episode(agent_trajectories, target_trajectory, total_reward, caught, axis_lim=5):
    """
    Plot the 2D trajectories of agents and the target with line thickness encoding the z-coordinate.

    Parameters:
    - agent_trajectories: dict mapping agent_id to list of [x, y, z] positions
    - target_trajectory: list of [x, y, z] positions for the target
    - total_reward: float, cumulative reward of the episode
    - caught: bool, whether the target was captured
    - axis_lim: float, limit for x and y axes (will be [-axis_lim, axis_lim])
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Colors per agent
    colors = {'agent_0': 'darkred', 'agent_1': 'orange'}
    
    # Plot each agent's trajectory
    for agent_id, traj in agent_trajectories.items():
        traj = np.array(traj)  # shape (T, 3)
        xy = traj[:, :2]
        zs = traj[:, 2]
        # Build segments between consecutive points
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        # Line widths proportional to average z of each segment
        widths = (zs[:-1] + zs[1:]) / 2.0
        lc = LineCollection(segments,
                            linewidths=widths,
                            colors=colors.get(agent_id, 'black'),
                            label=f"{agent_id}")
        ax.add_collection(lc)
        # Start and end markers
        ax.scatter(xy[0, 0], xy[0, 1], marker='s', color='darkred')
        ax.scatter(xy[-1, 0], xy[-1, 1], marker='^', color='darkred')
    
    # Plot target trajectory
    targ = np.array(target_trajectory)
    xy_t = targ[:, :2]
    zs_t = targ[:, 2]
    segments_t = np.stack([xy_t[:-1], xy_t[1:]], axis=1)
    widths_t = (zs_t[:-1] + zs_t[1:]) / 2.0
    lc_t = LineCollection(segments_t,
                          linewidths=widths_t,
                          colors='blue',
                          linestyles='dashed',
                          label="target")
    ax.add_collection(lc_t)
    ax.scatter(xy_t[0, 0], xy_t[0, 1], marker='s', color='blue')
    ax.scatter(xy_t[-1, 0], xy_t[-1, 1], marker='^', color='blue')
    
    # Build legend entries for start/end markers
    start_handle = mlines.Line2D([], [], color='gray', marker='s', linestyle='None', label='start')
    end_handle   = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', label='end')
   

    # Display reward and caught status
    ax.text(0.05, 0.95,
            f"Reward: {total_reward:.2f}\nCaught: {caught}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Formatting
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('PBT Episode Trajectories')
    # Legend with handles
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([start_handle, end_handle])
    ax.legend(handles=handles, labels=labels + ['start', 'end'], loc='upper right')
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()

def execute_feudal_episode(manager, workers, env):
    """
    Execute one feudal episode in render mode using the provided manager and workers.
    """
    obs_dict, _ = env.reset()
    agent_keys = sorted(workers.keys())
    agent_trajectories = {agent: [] for agent in agent_keys}
    target_trajectory = []
    total_reward = 0.0
    done = False
    last_target_pos = None
    last_info = {} 

    while not done:
    
        # record positions before acting
        for agent in agent_keys:
            agent_trajectories[agent].append(obs_dict[agent].copy())
        # target coords are always in obs_dict[agent_keys[0]][3:6]
        target_trajectory.append(obs_dict[agent_keys[0]][3:6].copy())
        # Render env
        env.render()
        pygame.event.pump()
        pygame.time.wait(100)
        
        # Manager step.
        pos_state = build_manager_global_state(obs_dict)
        current_target_pos = pos_state[-3:]  # last 3 entries
        if last_target_pos is None:
            tgt_vel = np.zeros(3, dtype=current_target_pos.dtype)
        else:
            tgt_vel = current_target_pos - last_target_pos
            
        last_target_pos = current_target_pos.copy()
        
        mgr_input = np.concatenate([pos_state, tgt_vel]) 
        global_state_tensor = torch.FloatTensor(mgr_input).unsqueeze(0).to(device)
        
        m_mean, m_std = manager.policy(global_state_tensor)
        m_dist = torch.distributions.Normal(m_mean, m_std)
        m_sample = m_dist.sample().squeeze(0)
        
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

        next_obs_dict, reward_dict, terminated, truncated, info = env.step(actions)
        total_reward += sum(reward_dict.values())
        last_info = info  # keep the info from the last step
        done = all(terminated[agent] or truncated[agent] for agent in agent_keys)
        obs_dict = next_obs_dict

    # Check for capture.

    caught = any(last_info.get("caught", {}).values())
    
    env.close()
    
    return caught, agent_trajectories, target_trajectory, total_reward

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
            size=5,
            target_speed=0.5
        )
        # Exectue and record logs
       
        caught, agent_traj, target_traj, total_reward = execute_feudal_episode(manager, workers, test_env)
        success_count += int(caught)
        
        # Show trajectories       
        visualize_episode(
            agent_trajectories=agent_traj,
            target_trajectory=target_traj,
            total_reward=total_reward,
            caught=caught,
            axis_lim=5
        )
        
    success_rate = success_count / test_rounds * 100
    print(f"\nTest Success Rate: {success_rate:.2f}% over {test_rounds} episodes.")

# =============================================================================
# Hyperparameter finetunning functions
# =============================================================================

def objective(trial):
    # Hyperparameter suggestions from Optuna.
    m_hidden_size = trial.suggest_int("m_hidden_size", 64, 256, step=64)
    w_hidden_size = trial.suggest_int("w_hidden_size", 64, 256, step=64)
    manager_lr = trial.suggest_float("manager_lr", 1e-5, 5e-4, log=True)
    worker_lr = trial.suggest_float("worker_lr", 1e-5, 5e-4, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.1)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    mu = trial.suggest_float("mu", 1e-5, 5e-4, log=True)
    lam = trial.suggest_float("lam", 0.90, 0.99)  # GAE lambda
    

    # Use a lower number of rounds to keep computation tractable.
    comm_rounds = 100        # Total communication rounds (max 1000 in full training)
    episodes_per_round = 50   # Episodes per communication round

    # Call your federated_feudal_training function with the suggested hyperparameters.
    round_rewards, global_manager, client_worker_agents, capture_rates = federated_feudal_training(
        num_clients=4,
        m_hidden_size=m_hidden_size,
        w_hidden_size=w_hidden_size,
        comm_rounds=comm_rounds,
        episodes_per_round=episodes_per_round,
        gamma=gamma,
        entropy_coef=entropy_coef,
        manager_lr=manager_lr,
        worker_lr=worker_lr,
        mu=mu,
        lam=lam
    )
    
    # Our objective is to maximize the final average reward, but since Optuna minimizes by default,
    # we return the negative of the reward from the last communication round.
    final_avg_reward = round_rewards[-1]
    return final_avg_reward



# =============================================================================
# Entry Point
# =============================================================================
training = True
testing = False
finetunning = False

# Hyperparameter definitions shared across training and testing.
m_hidden_size = 128
w_hidden_size = 256
manager_lr = 6.5e-05
worker_lr = 1.95e-05
entropy_coef = 0.012
gamma = 0.905 # 1 to remove lr decay
comm_rounds = 1000
episodes_per_round = 50
mu= 0.0002 # 0 to disable FedProx
gae_lam = 0.92 # 1 to disable GAE

tracker = EmissionsTracker(output_file="pbt_a2c_emissions.csv")
tracker.start()

if training:
    
    if __name__ == '__main__':       
        
        rewards, global_manager, client_worker_agents, capture_rates = federated_feudal_training(
            num_clients=4,
            m_hidden_size=m_hidden_size,
            w_hidden_size=w_hidden_size,
            comm_rounds=comm_rounds,
            episodes_per_round=episodes_per_round,
            gamma=gamma,
            entropy_coef=entropy_coef,
            manager_lr=manager_lr,
            worker_lr=worker_lr,
            mu=mu,
            lam=gae_lam            
        )
        
        # Compute average of the last 100 communication rounds
        # (If there are fewer than 100 rounds, use all rounds.)
        last_100_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        avg_last_100 = np.mean(last_100_rewards)
        
        # Define the file name
        file_name = "training_results.txt"
        
        # Check if the file exists and is non-empty
        file_empty = not os.path.exists(file_name) or os.path.getsize(file_name) == 0

        # Save hyperparameters and the average reward to a text file
        with open(file_name, "a", newline="") as f:
            # If the file is empty, write the header.
            if file_empty:
                header = ",".join([
                    "m_hidden_size",
                    "w_hidden_size",
                    "manager_lr",
                    "worker_lr",
                    "entropy_coef",
                    "gamma",
                    "comm_rounds",
                    "episodes_per_round",
                    "mu",
                    "gae_lam",
                    "avg_last_100"
                ])
                f.write(header + "\n")
            
            # Write the hyperparameter values and the average reward as a new comma-separated line.
            values = ",".join([
                str(m_hidden_size),
                str(w_hidden_size),
                str(manager_lr),
                str(worker_lr),
                str(entropy_coef),
                str(gamma),
                str(comm_rounds),
                str(episodes_per_round),
                str(mu),
                str(gae_lam),
                f"{avg_last_100:.2f}"
            ])
            f.write(values + "\n")
        
        print("\nHyperparameters and results appended to", file_name)
        
        # ------ Save the trained models -------
        # Save the global manager's policy and value networks.
        torch.save(global_manager.policy_net.state_dict(), 'global_manager_policy_pbt.pth')
        torch.save(global_manager.value_net.state_dict(), 'global_manager_value_pbt.pth')
        
        # Save the global worker model from one client (all clients are averaged).
        global_worker = list(client_worker_agents[0].values())[0]
        torch.save(global_worker.policy_net.state_dict(), 'global_worker_policy_pbt.pth')
        torch.save(global_worker.value_net.state_dict(), 'global_worker_value_pbt.pth')
        
        

if testing:
    
    # Reinitialize the manager with the same architecture/hyperparameters.
    num_agents = 2           # Two worker agents per client.
    manager_state_size = 3 * num_agents + 3 + 3
    goal_size = 3    
    global_manager = ManagerAgent(manager_state_size, goal_size, num_agents,
                                  xy_scale=5, z_scale=1.5, lr=manager_lr, hidden_size=m_hidden_size)
    
    # Load the saved state dictionaries for the manager.
    global_manager.policy_net.load_state_dict(torch.load('global_manager_policy_pbt.pth', map_location=device))
    global_manager.value_net.load_state_dict(torch.load('global_manager_value_pbt.pth', map_location=device))
    
    # Reinitialize a single worker agent.
    worker_state_size = 3 + goal_size  # Worker input dimension.
    action_size = 3
    
    global_worker = WorkerAgent(worker_state_size, action_size, lr=worker_lr, hidden_size=w_hidden_size)
    
    # Load the saved state dictionaries for the worker.
    global_worker.policy_net.load_state_dict(torch.load('global_worker_policy_pbt.pth', map_location=device))
    global_worker.value_net.load_state_dict(torch.load('global_worker_value_pbt.pth', map_location=device))
    
    # For testing, create a worker dictionary with 2 agents (they share the same global_worker).
    worker_dict = {
        'agent_0': global_worker,
        'agent_1': global_worker
    }
    
    # Create a single client list for testing.
    client_worker_agents = [worker_dict]
        
    # Optionally, test the trained model.
    test_feudrl_model(global_manager, client_worker_agents, test_rounds=20)

if finetunning:
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Adjust n_trials as needed.

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")  # Negate again to display the max reward.
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")    
    
emissions = tracker.stop()
print(f"Estimated CO2 emissions: {emissions} kg")