# -*- coding: utf-8 -*-
"""
Created on Tue May  6 21:11:43 2025

@author: mdbs1
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TRAINING RESULTS

# Load data

fedavg_ppo = np.load("training_metrics__fedavg_20250515_084649.npz")
pbt_a2c = np.load("training_metrics__pbt_20250504_231104.npz")
fedavg_a2c = np.load("training_metrics__fedavg_20250504_102317.npz")
pbt_basic = np.load("training_metrics__pbt_20250514_040249.npz")



# Extract metrics
fedavg_ppo_rewards = fedavg_ppo['round_rewards']
fedavg_ppo_capture = fedavg_ppo['capture_rates']
pbt_a2c_rewards = pbt_a2c['round_rewards']
pbt_a2c_capture = pbt_a2c['capture_rates']
fedavg_a2c_rewards = fedavg_a2c['round_rewards']
fedavg_a2c_capture = fedavg_a2c['capture_rates']
pbt_basic_rewards = pbt_basic['round_rewards']
pbt_basic_capture = pbt_basic['capture_rates']

def smooth(y, box_pts=10):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    pad_left = (len(y) - len(y_smooth)) // 2
    pad_right = len(y) - len(y_smooth) - pad_left
    return np.pad(y_smooth, (pad_left, pad_right), mode='edge')


# Apply smoothing
fedavg_ppo_rewards_smooth = smooth(fedavg_ppo_rewards)
pbt_a2c_rewards_smooth = smooth(pbt_a2c_rewards)
fedavg_a2c_rewards_smooth = smooth(fedavg_a2c_rewards)
pbt_basic_rewards_smooth = smooth(pbt_basic_rewards)

fedavg_ppo_capture_smooth = smooth(fedavg_ppo_capture)
pbt_a2c_capture_smooth = smooth(pbt_a2c_capture)
fedavg_a2c_capture_smooth = smooth(fedavg_a2c_capture)
pbt_basic_capture_smooth = smooth(pbt_basic_capture)

# Plot
plt.figure(figsize=(12, 6))
colors = plt.get_cmap("tab10").colors  # Color-blind friendly palette

# Round Rewards
plt.subplot(1, 2, 1)
plt.plot(fedavg_ppo_rewards_smooth, label='FedAvg PPO', color=colors[0])
plt.plot(pbt_a2c_rewards_smooth, label='PBT A2C', color=colors[1])
plt.plot(fedavg_a2c_rewards_smooth, label='FedAvg A2C', color=colors[2])
plt.plot(pbt_basic_rewards_smooth, label='PBT Basic', color=colors[3])
plt.title("Round Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

# Capture Rates
plt.subplot(1, 2, 2)
plt.plot(fedavg_ppo_capture_smooth, label='FedAvg PPO', color=colors[0])
plt.plot(pbt_a2c_capture_smooth, label='PBT A2C', color=colors[1])
plt.plot(fedavg_a2c_capture_smooth, label='FedAvg A2C', color=colors[2])
plt.plot(pbt_basic_capture_smooth, label='PBT Basic', color=colors[3])
plt.title("Capture Rates")
plt.xlabel("Episode")
plt.ylabel("Capture Rate")
plt.legend()

plt.tight_layout()
plt.show()

# TRAINING EMISSIONS


files = {
    'FedAvg A2C': 'fedavg_a2c_emissions_{device}.csv',  # replace 'gpu' with actual device if needed
    'PBT Basic': 'pbt_lite_emissions.csv',
    'FedAvg PPO': 'fedavg_ppo_emissions.csv',
    'PBT A2C': 'pbt_a2c_emissions.csv'
}

# Load all dataframes into a dictionary
dataframes = {name: pd.read_csv(path) for name, path in files.items()}


# Extract summary metrics for comparison
summary = []
for name, df in dataframes.items():
    total_emissions = df['emissions'].sum()
    total_energy = df['energy_consumed'].sum()
    total_duration = df['duration'].sum() / 3600

    summary.append({
        'Run': name,
        'Total Emissions (kg CO2)': total_emissions,
        'Total Energy (kWh)': total_energy,
        'Total Duration (h)': total_duration
    })

summary_df = pd.DataFrame(summary)

# Display summary table
print(summary_df)

# Use a consistent style
plt.style.use("ggplot")

# Load previous color scheme
colors = plt.get_cmap("tab10").colors
color_map = {
    'FedAvg PPO': colors[0],
    'PBT A2C': colors[1],
    'FedAvg A2C': colors[2],
    'PBT Basic': colors[3]
}

# Define figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Metrics to plot
metrics = [
    ('Total Emissions (kg CO2)', 'CO₂ Emissions (kg)'),
    ('Total Energy (kWh)', 'Energy Consumed (kWh)'),
    ('Total Duration (h)', 'Runtime Duration (h)')
]


for ax, (metric_key, title) in zip(axes, metrics):
    colors_in_order = [color_map[run] for run in summary_df['Run']]
    bars = ax.bar(summary_df['Run'], summary_df[metric_key], color=colors_in_order)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_key, fontsize=12)
    ax.set_xticklabels(summary_df['Run'], rotation=30, ha='right')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.suptitle("CodeCarbon Emission Metrics by Experiment", fontsize=16, fontweight='bold')
plt.show()
plt.style.use('default')