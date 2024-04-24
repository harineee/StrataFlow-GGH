# StrataFlow-GGH

StrataFlow aims to optimize the design of a network-on-chip (NOC) system within a silicon design environment. The goal is to achieve optimal performance metrics, including latency, bandwidth, buffer occupancy, and throttling frequency. This project utilizes a simulator setup and Reinforcement Learning (RL) techniques to derive an optimal NOC design. 'StrataFlow' is a portmanteau of the words 'strata' and 'data flow'.

The code simulates a network-on-chip (NOC) system's performance metrics like latency, bandwidth, and buffer occupancy. It reads simulated monitor output data from a CSV file and calculates average read latency and bandwidth based on the data. The code parses each row of the data to extract timestamps and transaction types, calculates the total bytes transferred assuming 32 bytes per write transaction, and determines the total cycles based on the timestamps. It then computes the average read latency by subtracting consecutive read timestamps and calculates the average bandwidth as the total bytes transferred divided by the total cycles. 

The RL technique chosen is Deep Q-Networks (DQN), to develop a robust design framework. This framework will derive optimal parameters for the NOC, considering factors like measured latency, bandwidth, buffer sizing, and throttling frequency. DQN is well-suited for problems where there is a large state space and actions have long-term consequences, which aligns with the optimization goals of the NOC design. 

Project Structure:
1. README.md: This file contains an overview of the project, its goals, approach, and project structure.
2. design.txt: This file contains detailed design specifications, architecture diagrams and any other design-related documentation.
3. pseudo.py: This Python file contains pseudocode or initial code snippets related to the project's simulation and optimization algorithms.
4. sim.csv: This file contains an example of the possible data within the simulator to illustrate the workings of this program.

To get started with this project:
1. Clone the repository to your local machine.
2. Review the README.md file for an overview of the project and its objectives.
3. Refer to design.txt for detailed design specifications, if available.
4. Open pseudo.py to explore the initial pseudocode or code snippets related to the simulation and optimization algorithms.
