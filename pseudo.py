import time
import random

f=1  #operating frequency of the system
w=32  #data width in bits

def bo(b_id):
    return random.randint(70, 100)  #random buffer occupancy percentage for demonstration

def ar(agent_t):
    return random.uniform(0.5, 1.0)  #random arbiter rate for demonstration

def pl():
    return random.choice([0, 1])  #random power limit threshold for demonstration

def rl():
    return random.randint(5, 15)  #random read latency in cycles for demonstration

def wl():
    return random.randint(10, 20)  #random write latency in cycles for demonstration

def bw(bt, cy):
    #bandwidth calculation in bytes per second
    return bt / (cy * f)

#main function to simulate the system
def simulate_system():
    buffer_ids = [1, 2, 3]  #example buffer IDs for demonstration
    agent_types = ["CPU", "IO"]  #example agent types for demonstration

    #simulate system operation
    while True:
        for b_id in buffer_ids:
            for agent_t in agent_types:
                buff_occ = bo(b_id)
                arb_rate = ar(agent_t)
                pow_limit = pl()

                read_lat = rl()
                write_lat = wl()

                read_bw = bw(w, read_lat)
                write_bw = bw(w, write_lat)

                #print simulation results
                print(f"Buffer ID: {b_id}, Agent Type: {agent_t}")
                print(f"Buffer Occupancy: {buff_occ}%")
                print(f"Arbiter Rate ({agent_t}): {arb_rate}")
                print(f"Power Limit Threshold: {pow_limit}")
                print(f"Read Latency: {read_lat} cycles")
                print(f"Write Latency: {write_lat} cycles")
                print(f"Read Bandwidth: {read_bw} bytes/sec")
                print(f"Write Bandwidth: {write_bw} bytes/sec")

        time.sleep(1)  #simulate waiting for the next cycle

#execute the sim
if __name__ == "__main__":
    simulate_system()
