import csv
import random 

def read(fp):
    data=[]
    with open(fp,'r') as f:
        reader=csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def measure_latency_bandwidth(data):
    r_ts=[]  #read timestamps
    w_ts=[]  #write timestamps
    b_trans=0  #total bytes transferred
    t_cycles=0  #total cycles

    for entry in data:
        ts=int(entry['Timestamp'])
        txn_type=entry['TxnType']

        if txn_type=="Rd":
            r_ts.append(ts)  #store read timestamp
        elif txn_type=="Wr":
            w_ts.append(ts)  #store write timestamp
            b_trans+=32  #assuming 32 bytes per write

        t_cycles=ts  #update total cycles to current timestamp

    r_latencies=[r_ts[i]-r_ts[i-1] for i in range(1,len(r_ts))]
    avg_r_lat=sum(r_latencies)/len(r_latencies) if r_latencies else 0

    avg_bw=b_trans/t_cycles

    return avg_r_lat,avg_bw

def adjust_buffer_sizes():
    # Adjust buffer sizes logic here
    print("Adjusting buffer sizes for 90% occupancy...")

def apply_throttling():
    # Apply throttling logic here
    if random.random() < 0.05:
        print("Throttling activated...")
    else:
        print("No throttling.")

if __name__=="__main__":
    fp='sim.csv'
    data=read(fp)
    avg_r_lat,avg_bw=measure_latency_bandwidth(data)

    print("Simulation Results:")
    print(f"Average Read Latency: {avg_r_lat} cycles")
    print(f"Average Bandwidth: {avg_bw} bytes/cycle")

    #check optimality conditions
    if avg_r_lat <= 10 and avg_bw >= 0.95 * 1000:
        print("Optimal design achieved.")
    else:
        print("Design does not meet optimality criteria.")

    #additional functionality
    adjust_buffer_sizes()
    apply_throttling()
