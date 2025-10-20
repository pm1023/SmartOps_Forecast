# =========================
# SMARTOPS FORECAST - PHASE 3
# DQN Agent for Multi-Product Inventory Optimization
# =========================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000):
        self.state_size = state_size          # Number of products (inventory state)
        self.action_size = action_size        # Number of products (order quantity actions)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma                    # Discount factor
        self.epsilon = epsilon                # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        
        # Build Q-network
        self.model = self._build_model()
    
    def _build_model(self):
        """Create a simple feedforward neural network"""
        model = Sequential([
            Dense(self.hidden_size, input_dim=self.state_size, activation='relu'),
            Dense(self.hidden_size, activation='relu'),
            Dense(self.action_size, activation='linear')  # Predict Q-values for each product
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy strategy"""
        if np.random.rand() <= self.epsilon:
            # Random action (exploration)
            return np.random.randint(0, 10, size=self.action_size)  # order 0-10 units
        # Predict Q-values (exploitation)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        # Use predicted Q-values as action
        return np.round(q_values[0]).astype(int)
    
    def replay(self, batch_size=32):
        """Train network using a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Bellman equation
                target += self.gamma * np.max(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0] = target  # update Q-value for the action taken
            
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
