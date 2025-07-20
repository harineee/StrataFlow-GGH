"""
This module defines the baseline structure for the StrataFlow framework, which applies
Deep Q-Networks (DQN) to optimize NoC design parameters by learning from simulated traffic data.

The agent seeks to minimize latency and energy consumption while maximizing bandwidth and
reducing packet loss and buffer pressure.
----------------------------------------------------------------------------------------------------------------
"""

import csv
import random
import numpy as np
import tensorflow as tf
from collections import deque
from typing import List, Tuple, Dict


# === [1] Data Handling === #
def read_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Reads the simulator log CSV and returns a list of transaction entries.

    Each entry contains:
        - Timestamp: int
        - TxnType: 'Rd' or 'Wr'
    """
    with open(file_path, 'r') as f:
        return list(csv.DictReader(f))


# === [2] NoC Metric Simulation === #
def measure_metrics(data: List[Dict[str, str]]) -> Tuple[float, float, int, int, int]:
    """
    Simulates performance metrics based on NoC transaction trace.

    Returns:
        avg_read_latency: float
        avg_bandwidth: float
        buffer_occupancy: int
        packet_loss: int
        energy_consumed: int
    """
    read_ts, total_bytes, buffer, loss, energy = [], 0, 0, 0, 0
    buffer_limit = 100
    last_time = 0

    for row in data:
        ts = int(row['Timestamp'])
        txn = row['TxnType']
        last_time = ts

        if txn == 'Rd':
            read_ts.append(ts)
        elif txn == 'Wr':
            total_bytes += 32
            buffer += 10
            energy += 5
            if buffer > buffer_limit:
                loss += 1
                buffer = buffer_limit

    read_latencies = [read_ts[i] - read_ts[i - 1] for i in range(1, len(read_ts))]
    avg_latency = sum(read_latencies) / len(read_latencies) if read_latencies else 0
    avg_bw = total_bytes / last_time if last_time else 0

    return avg_latency, avg_bw, buffer, loss, energy


# === [3] Deep Q-Network Agent === #
class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='mse')
        return model

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self.model.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int) -> None:
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_update = reward
            if not done:
                q_update += self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])
            q_values = self.model.predict(state, verbose=0)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# === [4] Training Loop === #
def train_noc_agent(data: List[Dict[str, str]], episodes: int = 100) -> None:
    state_size, action_size = 5, 3
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for ep in range(1, episodes + 1):
        # Initial state from metrics
        latency, bw, buf, loss, energy = measure_metrics(data)
        state = np.array([latency, bw, buf, loss, energy]).reshape(1, -1)

        for step in range(len(data)):
            action = agent.act(state)

            # Simulate effects of actions
            if action == 0:  # Reduce buffer pressure
                buf = max(buf - 10, 0)
            elif action == 1:  # Apply throttling
                energy += 2
            elif action == 2:  # Energy management
                energy = max(energy - 1, 0)

            # Re-measure state (simulate environment response)
            next_state = np.array([latency, bw, buf, loss, energy]).reshape(1, -1)

            # Define reward signal
            reward = (1 / (latency + 1e-5)) + (bw / 1000) - (loss * 0.2) - (energy * 0.01)
            done = step == len(data) - 1

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"[Episode {ep}/{episodes}] Latency={latency:.2f}, BW={bw:.2f}, Buffer={buf}, Loss={loss}, Energy={energy:.2f}, Epsilon={agent.epsilon:.3f}")


# === [5] Entry Point === #
if __name__ == "__main__":
    data = read_csv("sim.csv")
    train_noc_agent(data, episodes=100)
