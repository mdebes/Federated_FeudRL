# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:16:36 2025

@author: mdbs1
    
    Train agents without manager input

Feudal Reinforcement Learning implementation for the Catch environment based on Crazy RL https://github.com/ffelten/CrazyRL/tree/main
Here a single Manager network predicts an optimal target point (goal) that the
two Worker agents use (by concatenating it to their own observation) to choose actions.
We want to demonstrate that the manager agent predictions can improve the efficiency of catching the target.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import time
from crazy_rl.multi_agent.numpy.catch.catch import Catch
from crazy_rl.multi_agent.numpy.base_parallel_env import CLOSENESS_THRESHOLD

# -----------------------------
# Define Manager and Worker Networks.
# -----------------------------
class ManagerNetwork(nn.Module):
    def __init__(self, state_size, goal_size, num_agents, scale, hidden_size=128):
        super(ManagerNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        # Output a goal for each agent (each goal has goal_size dimensions)
        self.fc_mean = nn.Linear(hidden_size, num_agents * goal_size)
        self.fc_std = nn.Linear(hidden_size, num_agents * goal_size)
        self.num_agents = num_agents
        self.goal_size = goal_size
        self.scale = scale  # scale factor to match the environment's coordinate range

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        raw_mean = self.fc_mean(x)
        # Uniform scaling for all dimensions:
        mean = torch.tanh(raw_mean) * self.scale  
        std = F.softplus(self.fc_std(x)) + 1e-5  # to ensure positivity
        mean = mean.view(-1, self.num_agents, self.goal_size)
        std = std.view(-1, self.num_agents, self.goal_size)
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
    
# -----------------------------
# Define Value Networks (Critics)
# -----------------------------
class ManagerValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ManagerValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_value(x)  # Unbounded scalar output

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

# -----------------------------
# Define Feudal Agents: Manager and Worker.
# -----------------------------
class ManagerAgent:
    def __init__(self, state_size, goal_size, num_agents, scale, lr=5e-4):
        self.policy_net = ManagerNetwork(state_size, goal_size, num_agents, scale)
        self.value_net = ManagerValueNetwork(state_size)                   # ACTOR-CRITIC CHANGE
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # ACTOR-CRITIC CHANGE
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)    # ACTOR-CRITIC CHANGE

    def policy(self, x):
        return self.policy_net(x)
    
    def value(self, x):                 # ACTOR-CRITIC CHANGE
        return self.value_net(x)

# WorkerAgent now includes a value network and separate optimizers.
class WorkerAgent:
    def __init__(self, state_size, action_size, lr=5e-4):
        self.policy_net = WorkerNetwork(state_size, action_size)
        self.value_net = WorkerValueNetwork(state_size)                   # ACTOR-CRITIC CHANGE
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # ACTOR-CRITIC CHANGE
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)    # ACTOR-CRITIC CHANGE

    def policy(self, x):
        return self.policy_net(x)
    
    def value(self, x):                 # ACTOR-CRITIC CHANGE
        return self.value_net(x)

# -----------------------------
# Helper functions for state construction.
# -----------------------------

def extract_position(obs):
    """
    Returns the first three elements of an observation,
    which represent the agent’s own position.
    """
    return np.array(obs[:3])

def build_manager_global_state(obs_dict):
    """
    Build the global state by concatenating the position of each agent (3 numbers per agent)
    followed by the target's position (3 numbers). For example, with 2 agents, the state will have 9 dimensions.
    """
    agent_keys = sorted(obs_dict.keys())
    target = np.array(obs_dict[agent_keys[0]][3:6])
    features = []
    for agent in agent_keys:
        features.extend(extract_position(obs_dict[agent]))
    features.extend(target)
    return np.array(features)

def build_worker_state(obs_dict, current_agent_id):
    """
    For each worker, we use only its own position as the local state.
    """
    return extract_position(obs_dict[current_agent_id])

def discount_rewards(rewards, gamma):
    """
    Compute the discounted return for a sequence of rewards.
    """
    discounted = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted.insert(0, R)
    return discounted

def curriculum_weight(episode, curriculum_threshold=1000):
    # Returns a value between 0 and 1 that increases linearly over curriculum_duration episodes.
    return min(1.0, episode / curriculum_threshold)

# -----------------------------
# Environment parameter randomizers (unchanged).
# -----------------------------
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


def moving_average(data, window_size=100):
    """Compute the moving average over a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# -----------------------------
# Run one feudal episode.
#
# At the start of an episode the environment is reset, and the following steps occur repeatedly
# until termination:
#
# 1. The Manager receives a global state formed by concatenating all workers’ positions with the
#    true target position (extracted from the first agent’s observation) and outputs a goal
#    for each worker.
#
# 2. Each Worker constructs its input by concatenating its own local observation (its position)
#    with a goal. For episodes below a curriculum threshold, the worker uses the true target;
#    afterwards it uses the manager’s computed goal.
#
# 3. Each Worker samples an action from its policy using its constructed input.
#
# 4. The environment is stepped using the workers’ actions; each worker receives an extrinsic reward
#    from the environment.
#
# 5. The Manager and each Worker record their log probabilities, entropies, and the extrinsic rewards
#    (i.e. no separate intrinsic reward is computed) for later policy updates.
#
# -----------------------------

def train_feudal_episode(manager, workers, env, episode,
                         curriculum_threshold=1000, gamma=0.995, entropy_coef=0.01,
                         consistency_lambda=0.01, 
                         # New parameters for intrinsic reward:
                         intrinsic_coef=0.1,  # coefficient for intrinsic reward
                         debug_csv_filename="episode_log.csv"):
    """
    A modified feudal episode that decouples worker and manager updates.
    In addition, it incorporates:
      1. A per-step time penalty to encourage finishing the episode quickly.
      2. Intrinsic reward for how good worker follows manager goals
         in distance to the target.
    """
    # Initialize logs for manager and workers.
    manager_log_probs_list = []
    manager_rewards = []
    manager_entropies_list = []
    manager_values = []                     # ACTOR-CRITIC CHANGE: to store manager value estimates
    
    agent_keys = sorted(workers.keys())
    worker_log_probs = {agent: [] for agent in agent_keys}
    worker_rewards = {agent: [] for agent in agent_keys}
    worker_entropies = {agent: [] for agent in agent_keys}
    worker_values = {agent: [] for agent in agent_keys}  # ACTOR-CRITIC CHANGE: to store worker values
    
    # For consistency loss, store manager goals and observation history.
    manager_goals_list = []
    obs_history = []
    
    obs_dict, _ = env.reset()
    done = False
    total_episode_reward = 0

    while not done:
        obs_history.append(obs_dict)
        # -----------------------
        # Manager step: build global state and compute goal distributions.
        # -----------------------
        global_state = build_manager_global_state(obs_dict)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
        m_mean, m_std = manager.policy(global_state_tensor)
        m_value = manager.value(global_state_tensor)      # ACTOR-CRITIC CHANGE: get value estimate
        manager_values.append(m_value)                     # ACTOR-CRITIC CHANGE
        m_dist = torch.distributions.Normal(m_mean, m_std)
        m_sample = m_dist.sample().squeeze(0)  # shape: (num_workers, goal_size)
        m_log_prob = m_dist.log_prob(m_sample.unsqueeze(0)).sum(dim=2).squeeze(0)
        m_entropy = m_dist.entropy().sum(dim=2).squeeze(0)
        manager_log_probs_list.append(m_log_prob)
        manager_entropies_list.append(m_entropy)
        manager_goals_list.append(m_sample)
            
        # -----------------------
        # Worker step: each worker builds its state and samples an action.
        # -----------------------
        worker_goals = {}  # Dictionary to store the blended goal for each agent
        actions = {}
        
        for i, agent in enumerate(agent_keys):
            local_state = build_worker_state(obs_dict, agent)
            # Compute the curriculum weight (a value between 0 and 1)
            cw = curriculum_weight(episode, curriculum_threshold)
            # Extract the true target from the observation
            true_target = np.array(obs_dict[agent][3:6])
            # Manager's generated goal for this agent
            manager_goal = m_sample[i].detach().cpu().numpy()
            # Blend the targets based on the curriculum weight:
            worker_goal = (1 - cw) * true_target + cw * manager_goal
            worker_goals[agent] = worker_goal  # Save the blended goal for later use in reward computation.
            
            worker_input = np.concatenate([local_state, worker_goal])
            worker_input_tensor = torch.FloatTensor(worker_input).unsqueeze(0)
            w_value = workers[agent].value(worker_input_tensor)  # ACTOR-CRITIC CHANGE: get worker value
            worker_values[agent].append(w_value)                 # ACTOR-CRITIC CHANGE
            w_mean, w_std = workers[agent].policy(worker_input_tensor)
            w_dist = torch.distributions.Normal(w_mean, w_std)
            w_sample = w_dist.sample()
            action = w_sample.squeeze(0).detach().cpu().numpy()
            log_prob = w_dist.log_prob(w_sample).sum(dim=1)
            entropy = w_dist.entropy().sum(dim=1)
            worker_log_probs[agent].append(log_prob)
            worker_entropies[agent].append(entropy)
            actions[agent] = action
        
        # -----------------------
        # Step the environment.
        # -----------------------
        next_obs_dict, reward_dict, terminated, truncated, _ = env.step(actions)
        done = all(terminated[agent] or truncated[agent] for agent in agent_keys)
        
        # -----------------------
        # Compute intrinsic rewards.
        # -----------------------
        global_step_reward = 0  # for the manager, as the sum over agents.
        for agent in agent_keys:
            # Compute new distance after the step.
            worker_pos_next = extract_position(next_obs_dict[agent])
            # Intrinsic reward: penalize deviation from manager's goal.
            intrinsic_reward = -intrinsic_coef * np.linalg.norm(worker_pos_next - worker_goals[agent])
            # Combine the extrinsic reward with intrinsic reward.
            intrinsic_reward = reward_dict[agent] + intrinsic_reward
            worker_rewards[agent].append(intrinsic_reward)
            global_step_reward += intrinsic_reward
        
        manager_rewards.append(global_step_reward)
        total_episode_reward += global_step_reward
        obs_dict = next_obs_dict

    
    # -----------------------
    # Post-episode: Check termination conditions.
    # -----------------------
    first_key = list(obs_dict.keys())[0]
    target_pos = obs_dict[first_key][3:6]
    caught_train = any(
        np.linalg.norm(extract_position(obs_dict[agent]) - target_pos) < CLOSENESS_THRESHOLD
        for agent in obs_dict
    )

# -----------------------
    # Compute discounted returns for the manager.
    # -----------------------
    m_returns = discount_rewards(manager_rewards, gamma)
    m_returns = torch.FloatTensor(m_returns)
    m_values = torch.stack(manager_values, dim=0).view(-1)   # ACTOR-CRITIC CHANGE: stack recorded values
    m_advantages = m_returns - m_values.detach()        # ACTOR-CRITIC CHANGE: compute advantages


    manager_actor_loss = 0
    manager_critic_loss = 0                         # ACTOR-CRITIC CHANGE
    
    for t in range(len(m_returns)):
        step_log_prob = manager_log_probs_list[t].sum()
        step_entropy = manager_entropies_list[t].sum()
        manager_actor_loss += -step_log_prob * m_advantages[t] - entropy_coef * step_entropy
        manager_critic_loss += F.mse_loss(m_values[t].unsqueeze(0), m_returns[t].unsqueeze(0))
    value_coef = 0.5                             # ACTOR-CRITIC CHANGE: hyperparameter to balance losses
    manager_loss = manager_actor_loss + value_coef * manager_critic_loss

    # -----------------------
    # Compute discounted returns and loss for each worker.
    # -----------------------
    worker_loss_total = 0
    worker_actor_loss_total = 0                   # ACTOR-CRITIC CHANGE
    worker_critic_loss_total = 0                  # ACTOR-CRITIC CHANGE
    for agent in agent_keys:
        r = discount_rewards(worker_rewards[agent], gamma)
        r = torch.FloatTensor(r)
        w_values = torch.stack(worker_values[agent], dim=0).view(-1) # ACTOR-CRITIC CHANGE: stack values
        advantages = r - w_values.detach()                    # ACTOR-CRITIC CHANGE
        agent_actor_loss = 0
        agent_critic_loss = 0                 # ACTOR-CRITIC CHANGE
        for t in range(len(r)):
            agent_actor_loss += -worker_log_probs[agent][t] * advantages[t] - entropy_coef * worker_entropies[agent][t]
            agent_critic_loss += F.mse_loss(w_values[t].unsqueeze(0), r[t].unsqueeze(0))
        worker_actor_loss_total += agent_actor_loss
        worker_critic_loss_total += agent_critic_loss
    worker_loss_total = worker_actor_loss_total + value_coef * worker_critic_loss_total  # ACTOR-CRITIC CHANGE

    # -----------------------
    # Consistency loss (unchanged).
    # -----------------------
    consistency_loss = 0
    if episode >= curriculum_threshold:
        count = 0
        for t in range(len(manager_goals_list)):
            true_target = torch.FloatTensor(obs_history[t][agent_keys[0]][3:6])
            for idx, agent in enumerate(agent_keys):
                worker_pos = torch.FloatTensor(extract_position(obs_history[t][agent]))
                desired_goal = true_target - worker_pos
                predicted_goal = manager_goals_list[t][idx]
                consistency_loss += torch.nn.functional.mse_loss(predicted_goal, desired_goal)
                count += 1
        if count > 0:
            consistency_loss = consistency_loss / count

    # Global loss is the sum of manager loss, worker loss, and the consistency loss.
    global_loss = manager_loss + worker_loss_total + consistency_lambda * consistency_loss

    # -----------------------
    # Update the workers first.
    # -----------------------
    for agent in agent_keys:
        workers[agent].policy_optimizer.zero_grad()       # ACTOR-CRITIC CHANGE
        workers[agent].value_optimizer.zero_grad()          # ACTOR-CRITIC CHANGE
    worker_loss_total.backward(retain_graph=True)
    for agent in agent_keys:
        workers[agent].policy_optimizer.step()              # ACTOR-CRITIC CHANGE
        workers[agent].value_optimizer.step()               # ACTOR-CRITIC CHANGE
        
    # -----------------------
    # Then update the manager.
    # -----------------------
    manager.policy_optimizer.zero_grad()                  # ACTOR-CRITIC CHANGE
    manager.value_optimizer.zero_grad()                   # ACTOR-CRITIC CHANGE
    manager_loss.backward()
    manager.policy_optimizer.step()
    manager.value_optimizer.step()
    
    return total_episode_reward, manager_loss.item(), worker_loss_total.item(), global_loss.item(), caught_train

# -----------------------------
# Main feudal training loop.
# Now tracks episode rewards, manager loss, worker loss, and global loss.
# -----------------------------
def feudal_training(episodes=1000, curriculum_threshold=1000, gamma=0.995, entropy_coef=0.01, lr=5e-4):
    params = randomize_catch_parameters()
    env = Catch(
        drone_ids=np.arange(2),  # Dynamically adjust number of drones here
        render_mode=None,
        init_flying_pos=params["init_flying_pos"],
        init_target_location=params["init_target_location"],
        size=params["env_size"],
        target_speed=random_target_speed()
    )
    
    env.reset()
    num_workers = len(env.agents)
    # Manager receives global state: all worker positions + target position.
    manager_state_size = 3 * num_workers + 3  # (3 agent pos per worker) + 3 target pos
    goal_size = 3  # Predicted target position.
    # Each worker: local state = (own position) then concatenated with manager goal.
    worker_local_state_size = 3  # Only the agent's own position.
    worker_state_size = worker_local_state_size + goal_size  # 3 + 3 = 6.
    action_size = 3

    manager = ManagerAgent(manager_state_size, goal_size, num_agents=num_workers, scale=params["env_size"], lr=lr)
    # Worker agents share policy
    shared_worker = WorkerAgent(worker_state_size, action_size, lr=lr)
    workers = {agent_id: shared_worker for agent_id in env.agents}
    
    episode_rewards = []
    manager_losses = []
    worker_losses = []
    global_losses = []
    caught_training = []
    start_time = time.time()
    
    for ep in range(episodes):
        ep_reward, m_loss, w_loss, g_loss, target_caught = train_feudal_episode(manager, workers, env, ep,
                                                                 curriculum_threshold=curriculum_threshold,
                                                                 gamma=gamma,
                                                                 entropy_coef=entropy_coef)
        episode_rewards.append(ep_reward)
        manager_losses.append(m_loss)
        worker_losses.append(w_loss)
        global_losses.append(g_loss)
        caught_training.append(target_caught)
        
        if (ep+1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            elapsed = (time.time() - start_time) / 60
            print(f"\rEpisode {ep+1}/{episodes} completed! Avg Reward (last 100): {avg_reward:.2f} | Training time: {elapsed:.2f} minutes.", end="", flush=True)
    
    env.close()
    
    # Plot the progression of smoothed rewards.
    smoothed_rewards = moving_average(episode_rewards, window_size=1000)
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_rewards, label='Average Reward (1000 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Extrinsic Reward')
    plt.title('Feudal RL Training Progress (Smoothed over 1000 Episodes)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the progression of smoothed losses.
    smoothed_manager_losses = moving_average(manager_losses, window_size=1000)
    smoothed_worker_losses = moving_average(worker_losses, window_size=1000)
    smoothed_global_losses = moving_average(global_losses, window_size=1000)
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_manager_losses, label="Manager Loss (1000-episode avg)", color='blue')
    plt.plot(smoothed_worker_losses, label="Worker Loss (1000-episode avg)", color='red')
    plt.plot(smoothed_global_losses, label="Global Loss (1000-episode avg)", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss Progression during Feudal RL Training (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the progression of target captures.
    smoothed_rewards = moving_average(caught_training, window_size=1000)
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_rewards, label='Capture rate (1000 episodes)', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Capture Rate')
    plt.title('Feudal RL Training Progress (Smoothed over 1000 Episodes)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    return episode_rewards, manager, workers

# -----------------------------
# Execute a feudal episode in render mode during testing.
# -----------------------------
def execute_feudal_episode(manager, workers, env):
    """
    Execute one feudal episode in render mode.

    Parameters:
      manager: ManagerAgent object.
      workers: Dictionary mapping agent ids to WorkerAgent objects.
      env: The Catch environment (with render_mode="human").
      use_manager_goal: If True, workers use the manager's computed goal.
      intrinsic_coef: Coefficient for intrinsic reward (not used in testing).

    Returns:
      caught (bool): True if any agent comes close enough to the target.
      crash (bool): True if any two agents are too close (i.e., collide).
    """
    obs_dict, _ = env.reset()
    agent_keys = sorted(workers.keys())
    done = False

    while not done:
        # Render the current frame and process events to prevent crashes.
        env.render()
        pygame.event.pump()
        pygame.time.wait(100)  # Wait for 100 ms to allow event processing

        # -----------------------
        # Manager Step: Compute global state and manager's goals.
        # -----------------------
        global_state = build_manager_global_state(obs_dict)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
        m_mean, m_std = manager.policy(global_state_tensor)
        
        # Use the deterministic output (mean) for testing.
        # Access manager's network attributes (num_agents and goal_size) via policy_net.

        m_dist = torch.distributions.Normal(m_mean, m_std)
        m_sample = m_dist.sample().squeeze(0)  # shape: (num_workers, goal_size)                                                                          
        
        # Debug print: Check the range of manager goals.
        print("Manager goals:", m_sample)
        # Also print the manager's scale (which should be equal to the training environment size)
        print("Manager scale:", manager.policy_net.scale)

        actions = {}
        # -----------------------
        # Worker Step: Each worker computes its action based on its state and goal.
        # -----------------------
        for i, agent in enumerate(agent_keys):
            local_state = build_worker_state(obs_dict, agent)
            
            manager_goal = m_sample[i].detach().cpu().numpy()

            worker_goal = manager_goal

            worker_input = np.concatenate([local_state, worker_goal])
            worker_input_tensor = torch.FloatTensor(worker_input).unsqueeze(0)
            
            # Use deterministic action (mean) from the worker's policy.

            w_mean, w_std = workers[agent].policy(worker_input_tensor)
            w_dist = torch.distributions.Normal(w_mean, w_std)
            w_sample = w_dist.sample()
            action = w_sample.squeeze(0).detach().cpu().numpy()
            
            actions[agent] = action

        # -----------------------
        # Step the environment.
        # -----------------------
        next_obs_dict, reward_dict, terminated, truncated, _ = env.step(actions)
        done = all(terminated[agent] or truncated[agent] for agent in agent_keys)
        
        obs_dict = next_obs_dict

    # -----------------------
    # Post-episode: Check termination conditions.
    # -----------------------
    first_key = list(obs_dict.keys())[0]
    target_pos = obs_dict[first_key][3:6]
    caught = any(
        np.linalg.norm(extract_position(obs_dict[agent]) - target_pos) < CLOSENESS_THRESHOLD
        for agent in obs_dict
    )

    crash = False
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

# -----------------------------
# Test the trained feudal model.
# -----------------------------
def test_feudal_model(manager, workers, episodes=10):
    target_caught = []
    agent_crash = []
    
    for i in range(episodes):
        params = randomize_catch_parameters()
        test_env = Catch(
            drone_ids=np.array([0, 1]),
            render_mode="human",
            init_flying_pos=params["init_flying_pos"],
            init_target_location=params["init_target_location"],
            size=params["env_size"],
            target_speed=random_target_speed()
        )
        caught, crash = execute_feudal_episode(manager, workers, test_env)
        target_caught.append(caught)
        agent_crash.append(crash)
        
    success = sum(target_caught)
    failure = sum(agent_crash)
    success_rate = success / episodes
    failure_rate = failure / episodes
        
    print(f"\rSuccess rate: {success_rate} Crash rate: {failure_rate} Number of episodes: {episodes}.", end="", flush=True)

# -----------------------------
# Entry point: run feudal training and plot progress.
# -----------------------------
if __name__ == '__main__':
    lr = 1e-5 #5e-4 #1e-5
    entropy_coef = 0.1
    gamma = 0.95
    episodes = 40000
    curriculum_threshold = 10000
    rewards, manager, workers = feudal_training(episodes=episodes,
                                                curriculum_threshold=curriculum_threshold,
                                                gamma=gamma,
                                                entropy_coef=entropy_coef,
                                                lr=lr)
    
    test_feudal_model(manager, workers, episodes=10)
