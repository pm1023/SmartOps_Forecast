# =========================
# SMARTOPS FORECAST - PHASE 4
# Train Multi-Product DQN Agent
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.rl_environment import MultiProductInventoryEnv
from src.rl_agent import DQNAgent
import os

# ------------------------------
# 1️⃣ Load forecasted demand (simulated)
# ------------------------------
df = pd.read_csv("data/simulated_sales.csv", parse_dates=['Date'])
products = df['Product'].unique()
num_products = len(products)
num_days = df['Date'].nunique()

# Reshape demand into [num_days, num_products]
demand_data = df.pivot(index='Date', columns='Product', values='Units_Sold').fillna(0).values.astype(int)

# ------------------------------
# 2️⃣ Initialize environment
# ------------------------------
env = MultiProductInventoryEnv(demand_data=demand_data, max_inventory=50)

state_size = env.num_products
action_size = env.num_products  # each product has an order quantity

agent = DQNAgent(state_size, action_size)

# ------------------------------
# 3️⃣ Training parameters
# ------------------------------
episodes = 50
batch_size = 32
all_rewards = []

# ------------------------------
# 4️⃣ Training loop
# ------------------------------
for e in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Agent selects action (order quantities)
        action = agent.act(state)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Store experience in replay memory
        agent.remember(state, action, reward, next_state, done)
        
        # Learn from experience
        agent.replay(batch_size)
        
        state = next_state
        total_reward += reward
    
    all_rewards.append(total_reward)
    print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

# ------------------------------
# 5️⃣ Plot total rewards
# ------------------------------
os.makedirs("outputs/rl", exist_ok=True)

plt.figure(figsize=(10,5))
plt.plot(range(1, episodes+1), all_rewards, marker='o')
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward (Negative Cost)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/rl/total_rewards.png")
plt.show()
