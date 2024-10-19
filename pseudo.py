import csv
import random
import numpy as np
import tensorflow as tf
from collections import deque

# Read and process data function
def read(fp):
    data = []
    with open(fp, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def measure_metrics(data):
    r_ts = []  # read timestamps
    w_ts = []  # write timestamps
    b_trans = 0  # total bytes transferred
    t_cycles = 0  # total cycles
    buffer_occupancy = 0  # buffer usage percentage
    throttling_count = 0  # number of times throttling applied
    energy_consumed = 0  # energy consumption in arbitrary units
    packet_loss = 0  # count of dropped packets due to buffer overflow
    buffer_size = 100  # buffer size limit (can be modified)

    for entry in data:
        ts = int(entry['Timestamp'])
        txn_type = entry['TxnType']

        if txn_type == "Rd":
            r_ts.append(ts)  # store read timestamp
        elif txn_type == "Wr":
            w_ts.append(ts)  # store write timestamp
            b_trans += 32  # assuming 32 bytes per write
            buffer_occupancy += 10  # simulate buffer filling

            # Simulate buffer overflow
            if buffer_occupancy > buffer_size:
                packet_loss += 1
                buffer_occupancy = buffer_size  # cap the buffer occupancy

            # Energy consumption based on transaction type
            energy_consumed += 5  # example energy cost for write operation

        t_cycles = ts  # update total cycles to current timestamp

    # Calculate latency and bandwidth
    r_latencies = [r_ts[i] - r_ts[i - 1] for i in range(1, len(r_ts))]
    avg_r_lat = sum(r_latencies) / len(r_latencies) if r_latencies else 0
    avg_bw = b_trans / t_cycles if t_cycles != 0 else 0

    return avg_r_lat, avg_bw, buffer_occupancy, throttling_count, packet_loss, energy_consumed

# Define DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define environment interaction
def run_simulation(data, agent, episodes=1000):
    state_size = 5  # (Latency, Bandwidth, Buffer Occupancy, Throttling Count, Packet Loss)
    action_size = 3  # (Adjust buffer, Apply throttling, Manage energy)
    batch_size = 32

    for e in range(episodes):
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state (latency, bandwidth, etc.)
        state = np.reshape(state, [1, state_size])

        for step in range(len(data)):
            # Measure current system metrics
            avg_r_lat, avg_bw, buffer_occupancy, throttling_count, packet_loss, energy_consumed = measure_metrics(data)

            action = agent.act(state)  # Agent chooses an action
            
            # Apply the chosen action
            if action == 0:
                buffer_occupancy -= 10  # Reduce buffer usage
            elif action == 1:
                throttling_count += 1  # Simulate throttling event
                energy_consumed += 2  # Increase energy due to throttling
            elif action == 2:
                energy_consumed -= 1  # Simulate energy management action

            next_state = np.array([avg_r_lat, avg_bw, buffer_occupancy, throttling_count, packet_loss])
            next_state = np.reshape(next_state, [1, state_size])

            # Reward: higher for low latency, high bandwidth, low packet loss, and minimal energy consumption
            reward = (1.0 / (avg_r_lat + 1.0) + avg_bw / 1000.0) - (packet_loss * 0.1 + energy_consumed * 0.01)
            done = step == len(data) - 1

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"Episode {e+1}/{episodes}, Avg Latency: {avg_r_lat}, Avg Bandwidth: {avg_bw}, Buffer: {buffer_occupancy}%, Packet Loss: {packet_loss}, Energy: {energy_consumed}")

if __name__ == "__main__":
    fp = 'sim.csv'
    data = read(fp)
    
    state_size = 5  # Latency, Bandwidth, Buffer Occupancy, Throttling Count, Packet Loss
    action_size = 3  # Adjust buffer, Apply throttling, Manage energy
    agent = DQNAgent(state_size, action_size)

    run_simulation(data, agent)
